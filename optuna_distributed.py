# pyright: basic

import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np
import optuna
import psutil
from joblib import Parallel, delayed
from optuna.samplers import GridSampler
from sklearn.metrics import check_scoring
from sklearn.model_selection import KFold, cross_val_score


class PostgresManager:
    def __init__(self, location, interface, port=None, database="managed") -> None:
        if (master := Path(location).joinpath("postmaster.pid")).exists():
            pid, _, _, port, _, address, _, _ = master.read_text().splitlines()

            self.pid = None  # int(pid)
        else:
            address = psutil.net_if_addrs()[interface][0].address
            with socket.socket() as sock:
                sock.bind((address, 0))
                _, port = sock.getsockname()

            self.pid = self.start_postgres(location, address, port, database).pid

        self.port = port
        self.address = address
        self.database = database

    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.address}:{self.port}/{self.database}"

    @staticmethod
    def start_postgres(location, address, port, database) -> subprocess.Popen:
        if not (path := Path(location)).exists():
            subprocess.run(["initdb", "-D", path.as_posix()])
            subprocess.run(
                ["postgres", "--single", "-D", path.as_posix(), "postgres"],
                input=f"create database {database};".encode("utf-8"),
            )
            subprocess.run(
                ["postgres", "--single", "-D", path.as_posix(), "postgres"],
                input=f"alter system set max_connections to '1000';".encode("utf-8"),
            )
            Path(location).joinpath("pg_hba.conf").write_text(
                "\n".join(
                    [
                        "host  all all 0.0.0.0/0 trust",
                        "host  all all ::/0      trust",
                        "local all all           trust",
                    ]
                )
            )

        return subprocess.Popen(
            ["postgres", "-h", address, "-p", f"{port}", "-D", path.as_posix()],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )

    def close(self) -> None:
        if self.pid:
            psutil.Process(self.pid).terminate()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, type, value, traceback) -> None:
        self.close()


@dataclass
class GridSearchResults:
    best_params_: dict[str, list[Any]]
    best_estimator_: Any
    score_: float | None


@dataclass
class InnerGridSearchResults:
    best_params_: dict[str, list[Any]]
    best_estimator_: Any
    inner_score_: float | None
    outer_score_: float | None
    training_set_: tuple[np.ndarray, np.ndarray] | None
    test_set_: tuple[np.ndarray, np.ndarray] | None


@dataclass
class OuterGridSearchResults:
    best_params_: dict[str, list[Any]]
    best_estimator_: Any
    score_: float | None
    inner_: list[InnerGridSearchResults]


class Distributed:
    def __init__(self, database: PostgresManager) -> None:
        self.connection_string = database.connection_string

    def nested_grid_search(
        self,
        id,
        model,
        params,
        X,
        y,
        scoring=None,
        inner_cv=None,
        outer_cv=None,
        n_jobs=None,
    ):
        outer_cv = outer_cv or KFold()
        scorer = check_scoring(model, scoring)

        def inner_grid_search(fold, train_idx, test_idx):
            cloned_model = sklearn.base.clone(model)
            results = self.grid_search(
                id=f"{id}_fold{fold}",
                model=cloned_model,
                params=params,
                X=X[train_idx],
                y=y[train_idx],
                scoring=scoring,
                cv=inner_cv,
                refit=False,
            )

            cloned_model.set_params(**results.best_params_)  # type:ignore
            cloned_model.fit(X[train_idx], y[train_idx])  # type:ignore

            return InnerGridSearchResults(
                best_params_=results.best_params_,
                best_estimator_=cloned_model,
                inner_score_=results.score_,
                outer_score_=scorer(cloned_model, X[test_idx], y[test_idx]),  # type:ignore
                test_set_=(X[test_idx], y[test_idx]),
                training_set_=(X[train_idx], y[train_idx]),
            )

        results: list[InnerGridSearchResults] = Parallel(n_jobs=n_jobs)(
            delayed(inner_grid_search)(fold, train_idx, test_idx)
            for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y))
        )  # type:ignore

        all_scores = [result.outer_score_ for result in results]
        all_params = [result.best_params_ for result in results]
        most_common_params = max(all_params, key=all_params.count)

        model.set_params(**most_common_params)
        model.fit(X, y)

        return OuterGridSearchResults(
            best_params_=most_common_params,
            best_estimator_=model,
            score_=np.mean(all_scores),  # type:ignore
            inner_=results,
        )

    def grid_search(
        self, id, model, params, X, y, scoring=None, cv=None, refit=True
    ) -> GridSearchResults:
        sampler = GridSampler(params, seed=0)
        study = optuna.create_study(
            study_name=id,
            storage=self.connection_string,
            sampler=sampler,
            direction="maximize",
            load_if_exists=True,
        )
        cv = cv or KFold()

        def objective(trial: optuna.Trial) -> float:
            model.set_params(
                **{
                    key: trial.suggest_categorical(name=key, choices=values)
                    for key, values in params.items()
                }
            )

            return cross_val_score(model, X, y, cv=cv, scoring=scoring).mean()

        study.optimize(objective)

        best_trial = get_deterministic_best_trial(study) or study.best_trial

        if refit:
            model.set_params(**study.best_params)
            model.fit(X, y)

        return GridSearchResults(
            best_params_=best_trial.params,
            best_estimator_=model,
            score_=best_trial.value,
        )


def get_deterministic_best_trial(
    study: optuna.Study,
) -> optuna.trial.FrozenTrial | None:
    best_trials = [t for t in study.trials if t.value == study.best_value]
    sorted_trials = sorted(best_trials, key=lambda t: tuple(t.params.keys()))

    return sorted_trials[0] if len(sorted_trials) > 0 else None


import sklearn
import sklearn.datasets
from sklearn.svm import SVC


def main():
    X, y = sklearn.datasets.load_iris(return_X_y=True)

    with PostgresManager("db", interface="lo") as database:
        model = SVC()
        params = {"C": [0.1, 1, 2, 4, 8, 10], "gamma": [0.1, 0.2, 0.4, 0.8]}

        for summary in optuna.study.get_all_study_summaries(
            storage=database.connection_string
        ):
            optuna.delete_study(
                study_name=summary.study_name, storage=database.connection_string
            )

        result = Distributed(database).nested_grid_search(
            "test",
            model,
            params,
            X,
            y,
            inner_cv=KFold(shuffle=True, random_state=0),
            outer_cv=KFold(shuffle=True, random_state=0),
            n_jobs=1,
        )

        print(result.score_)

        # for result in Parallel(n_jobs=1)(
        #     delayed(Distributed(database).grid_search)(
        #         f"test_{i}",
        #         model,
        #         params,
        #         X,
        #         y,
        #         cv=KFold(shuffle=True, random_state=0),
        #     )
        #     for i in range(100)
        # ):
        #     print(result.best_params_)

        # subprocess.run(["optuna-dashboard", database.connection_string])


if __name__ == "__main__":
    main()
