import pytest

from polymarket_bot.config import ConfigError
from polymarket_bot.database import MySQLConfig


def test_mysql_config_defaults_to_local_database() -> None:
    config = MySQLConfig.from_env({})

    assert config.host == "127.0.0.1"
    assert config.port == 3306
    assert config.database == "polymarket_bot"


def test_mysql_database_rejects_sql_identifiers() -> None:
    with pytest.raises(ConfigError):
        MySQLConfig.from_env({"MYSQL_DATABASE": "bad-name`; DROP DATABASE mysql"})
