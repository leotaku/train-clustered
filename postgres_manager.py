import socket
import subprocess
from pathlib import Path
from typing import Self

import psutil


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
