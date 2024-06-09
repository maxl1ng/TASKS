from __future__ import annotations

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score


import datetime
from typing import Optional, Union, Iterable, Sequence, cast, Any
import weakref


class InvalidClientError(ValueError):
    """Файл входных данных имеет недопустимое значение"""


# Создание суперкласса
class Client:

    # Определение контруктора, параметр self является ссылкой на текущий экземпляр класса
    def __init__(
        self,
        seniority: int,
        home: int,
        age: int,
        marital: int,
        records: int,
        expenses: int,
        assets: int,
        amount: int,
        price: int,
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

    # Удобный вывод
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"seniority={self.seniority},"
            f"home={self.home},"
            f"age={self.age},"
            f"marital={self.marital},"
            f"records={self.records},"
            f"expenses={self.assets},"
            f"amount={self.amount},"
            f"price={self.price}"
            f")"
        )


# Класс известного клиента
class KnownClient(Client):
    def __init__(
        self,
        status: int,
        seniority: int,
        home: int,
        age: int,
        marital: int,
        records: int,
        expenses: int,
        assets: int,
        amount: int,
        price: int,
    ) -> None:
        super().__init__(
            seniority=seniority,
            home=home,
            age=age,
            marital=marital,
            records=records,
            expenses=expenses,
            assets=assets,
            amount=amount,
            price=price,
        )
        self.status = status

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"seniority={self.seniority},"
            f"home={self.home},"
            f"age={self.age},"
            f"marital={self.marital},"
            f"records={self.records},"
            f"expenses={self.assets},"
            f"amount={self.amount},"
            f"price={self.price},"
            f"status={self.status!r},"
            f")"
        )

    @classmethod
    def from_dict(cls, row: dict[str, str]) -> "KnownClient":
        if row["status"] not in {"0", "1", "2"}:
            raise InvalidClientError(f"Invalid status in {row!r}")
        try:
            return cls(
                seniority=int(row["seniority"]),
                home=int(row["home"]),
                age=int(row["age"]),
                marital=int(row["marital"]),
                records=int(row["records"]),
                expenses=int(row["expenses"]),
                assets=int(row["assets"]),
                amount=int(row["amount"]),
                price=int(row["price"]),
                status=int(row["status"]),
            )
        except ValueError as exeption:
            raise InvalidClientError(f"Invalid {row!r}")


# Класс известного клиента для тренировочных данных
class TrainingKnownClient(KnownClient):

    @classmethod
    def from_dict(cls, row: dict[str, str]) -> "TrainingKnownClient":
        return cast(TrainingKnownClient, super().from_dict(row))


# Класс известного клиента для тестовых данных
class TestingKnownClient(KnownClient):
    def __init__(
        self,
        status: int,
        seniority: int,
        home: int,
        age: int,
        marital: int,
        records: int,
        expenses: int,
        assets: int,
        amount: int,
        price: int,
        classification: Optional[int] = None,
    ) -> None:
        super().__init__(
            status,
            seniority,
            home,
            age,
            marital,
            records,
            expenses,
            assets,
            amount,
            price,
        )
        self.classification = classification

    # Сравнение реального значения с классифицированным
    def mathces(self) -> bool:
        return self.status == self.classification

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"seniority={self.seniority},"
            f"home={self.home},"
            f"age={self.age},"
            f"marital={self.marital},"
            f"records={self.records},"
            f"expenses={self.assets},"
            f"amount={self.amount},"
            f"price={self.price},"
            f"status={self.status!r},"
            f"classification={self.classification!r}"
            f")"
        )

    @classmethod
    def from_dict(cls, row: dict[str, str]) -> "TestingKnownClient":
        return cast(TestingKnownClient, super().from_dict(row))


# Класс неизвестного клиента (нуждается в классификации)
class UnknownClient(Client):

    @classmethod
    def from_dict(cls, row: dict[str, str]) -> "UnknownClient":
        if set(row.keys()) != {
            "seniority",
            "home",
            "age",
            "marital",
            "records",
            "assets",
            "amount",
            "price",
        }:
            raise InvalidClientError(f"Invalid fields in {row!r}")
        try:
            return cls(
                seniority=int(row["seniority"]),
                home=int(row["home"]),
                age=int(row["age"]),
                marital=int(row["marital"]),
                records=int(row["records"]),
                expenses=int(row["expenses"]),
                assets=int(row["assets"]),
                amount=int(row["amount"]),
                price=int(row["price"]),
            )
        except (KeyError, ValueError):
            raise InvalidClientError(f"Invalid {row!r}")


# Класс классифицированного клиента
class ClassifiedClient(Client):
    def __init__(self, classification: Optional[int], client: UnknownClient) -> None:
        super().__init__(
            seniority=client.seniority,
            home=client.home,
            age=client.age,
            marital=client.marital,
            records=client.records,
            expenses=client.expenses,
            assets=client.assets,
            amount=client.amount,
            price=client.price,
        )
        self.classification = classification

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"seniority={self.seniority},"
            f"home={self.home},"
            f"age={self.age},"
            f"marital={self.marital},"
            f"records={self.records},"
            f"expenses={self.assets},"
            f"amount={self.amount},"
            f"price={self.price},"
            f"classification={self.classification!r}"
            f")"
        )


# Гиперпараметр. Класс для определения параметров модели.
class Hyperparameter:
    def __init__(
        self, max_depth: int, min_sample_size: int, training: "TrainingData"
    ) -> None:
        self.max_depth = max_depth
        self.min_leaf_size = min_sample_size
        self.data: weakref.ReferenceType["TrainingData"] = weakref.ref(training)
        self.quality: float

    # Обучение и тест модели.
    def test(self) -> None:
        training_data: Optional["TrainingData"] = self.data()
        if not training_data:
            raise RuntimeError("Broken Weak Reference")
        test_data = training_data.testing
        x_test = TrainingData.get_list_clients(test_data)
        y_test = TrainingData.get_statuses_clients(test_data)
        y_predict = self.classify_list(test_data)
        self.quality = roc_auc_score(y_test, y_predict)
        for i in range(len(y_predict)):
            test_data[i].classification = y_predict[i]

    # Классификация
    def classify_list(
        self, clients: Sequence[Union[UnknownClient, TestingKnownClient]]
    ) -> list[Any]:
        training_data = self.data()
        if not training_data:
            raise RuntimeError("No training object")
        x_predict = TrainingData.get_list_clients(clients)
        x_train = TrainingData.get_list_clients(training_data.training)
        y_train = TrainingData.get_statuses_clients(training_data.training)

        classifier = DecisionTreeClassifier()
        classifier = classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_predict).tolist()
        return [y_pred]


# Класс тренировочных данных
class TrainingData:
    def __init__(self, name: str) -> None:
        self.name = name
        self.uploaded: datetime.datetime
        self.tested: datetime.datetime
        self.training: list[TrainingKnownClient] = []
        self.testing: list[TestingKnownClient] = []
        self.tuning: list[Hyperparameter] = []

    # Загрузка данных с помощью словаря
    def load(self, raw_data_soruce: Iterable[dict[str, str]]) -> None:

        for n, row in enumerate(raw_data_soruce):
            try:
                if n % 5 == 0:
                    testing_client = TestingKnownClient.from_dict(row)
                    self.testing.append(testing_client)
                else:
                    training_data = TrainingKnownClient.from_dict(row)
                    self.training.append(training_data)
            except InvalidClientError as exception:
                print(f"Row: {n + 1} {exception}")
                return
        self.uploaded = datetime.datetime.now(tz=datetime.timezone.utc)

    def test(self, parameter: Hyperparameter) -> None:

        parameter.test()
        self.tuning.append(parameter)
        self.tested = datetime.datetime.now(tz=datetime.timezone.utc)

    def classify(self, parameter: Hyperparameter, client: Client) -> Client:

        classification = parameter.classify_list([client])[0], client = client
        return client

    # Получение списка свойств клиентов обучения
    @staticmethod
    def get_list_clients(clients: Sequence[Client]) -> list[list[int]]:
        return [
            [
                client.seniority,
                client.home,
                client.age,
                client.marital,
                client.records,
                client.expenses,
                client.assets,
                client.amount,
                client.price,
            ]
            for client in clients
        ]

    # Получение списка статусов клиентов
    @staticmethod
    def get_statuses_clients(clients: Sequence[KnownClient]) -> list[int]:
        return [client.status for client in clients]

    # Получение списка свойств клиента для классификации
    @staticmethod
    def get_client_as_list(client: Client) -> list[list[int]]:
        return [
            [
                client.seniority,
                client.home,
                client.age,
                client.marital,
                client.records,
                client.expenses,
                client.assets,
                client.amount,
                client.price,
            ]
        ]
