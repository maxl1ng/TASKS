from collections.abc import Iterator
from __future__ import annotations
import datetime
from typing import Optional, Union, Iterable

class TrainingData:
    def __init__(self, name, uploaded, tested) -> None:
        pass

class Client:
    def __init__(self,
                 seniority: int,
                 home: int,
                 age: int,
                 marital: int,
                 records: int,
                 expenses: int,
                 assets: int,
                 amount: int,
                 price: int,
                 status: Optional[str] = None
                 ) -> None:
        self.seniority = seniority
        self.home = home
        self.age = age
        self.marital = marital
        self.records = records
        self.expenses = expenses
        self.assets = assets
        self.amount = amount
        self.price = price
        self.status = status
        self.classification: Optional[str] = None


    def __repr__(self) -> str:
        if self.status is None:
            known_unknown = "UnknownStatus"
        else:
            known_unknown = "KnownStatus"
        if self.classification is None:
            classification = ""
        else:
            classification = f", classification={self.classification!r}"
        return (
            f"{known_unknown}("
            f"seniority={self.seniority}"
            f"home={self.home}"
            f"age={self.age}"
            f"marital={self.marital}"
            f"records={self.records}"
            f"expenses={self.assets}"
            f"amount={self.amount}"
            f"price={self.price}"
            f"classification={classification}"
            f")"
        )
    def classify(self, classification: str) -> None:
        self.classification = classification

    def matches(self) -> bool:
        return self.status == self.classification

class Hyperparameter:
    def __init__(self, max_depth: int, min_leaf_size: int, training: "TrainingData") -> None:
        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size
        self.data: TrainingData = training
        self.quality: float

    def test(self) -> None:
        """
            Проверка на тестовом наборе данных
        """
        pass_count, fail_count = 0, 0
        for client in self.data.testing:
            if client.matches():
                pass_count += 1
            else:
                fail_count += 1
        self.quality = pass_count / (pass_count + fail_count)

    def classify(self, client: Client) -> str:
        """TODO: вставить модель дерева решений

        Args:
            client (Client): _description_

        Returns:
            str: _description_
        """
        return ""

class TrainingData:
    def __init__(self, name: str) -> None:
        self.name = name
        self.uploaded: datetime.datetime
        self.tested: datetime.datetime
        self.training: list[Client] = []
        self.testing: list[Client] = []
        self.tuning: list[Hyperparameter] = []

    def load(self, raw_data_soruce: Iterable[dict[str, str]]) -> None:
        """
        Загрузка и разбитие исходных данных

        Args:
            raw_data_soruce (Iterable[dict[str, str]]): Источник данных
        """

        for n, row in enumerate(raw_data_soruce):
            client = Client(
                seniority = int(row["seniority"]),
                home = int(row["home"]),
                age = int(row["age"]),
                marital = int(row["marital"]),
                records = int(row["records"]),
                expenses = int(row["expenses"]),
                assets = int(row["assets"]),
                amount = int(row["amount"]),
                price = int(row["price"]),
                status = row["status"]
            )

            # Разбивка выборки: каждая 5-я строка в текстовую
            if n % 5 == 0:
                self.testing.append(client)
            else:
                self.training.append(client)
        self.uploaded = datetime.date.now(tz=datetime.timezone.utc)

    def test(self, parameter: Hyperparameter) -> None:
        """

        Args:
            parameter (Hyperparameter): Гиперпараметры
        """


        parameter.test()
        self.tuning.append(parameter)
        self.tested = datetime.datetime.now(tz=datetime.timezone.utc)


    # мб статический метод
    def classify(self, parameter: Hyperparameter, client: Client) -> Client:
        """

        Args:
            parameter (Hyperparameter): _description_
            client (Client): _description_

        Returns:
            Client: _description_
        """

        # Классифицируем
        classification = parameter.classify(client)
        client.classify(classification)
        return client