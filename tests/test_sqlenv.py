"""SQLite text→SQL environment: the read-only jail (safety) + reward/shaper.

The model's SQL is untrusted, so the security-critical property is that NOTHING
but a read can execute. These tests assert writes/DDL/attach/stacked statements
all fail AND leave the database unchanged.
"""

import sqlite3

import pytest

from sdft.online import sqlenv
from sdft.online.reward import get_reward_fn, get_shaper


@pytest.fixture()
def db(tmp_path):
    path = sqlenv.build_db(str(tmp_path / "store.db"))
    sqlenv.set_db(path)
    yield path
    sqlenv.set_db(None)


def _count(path, table):
    con = sqlite3.connect(path)
    try:
        return con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        con.close()


class TestReadOnlyJail:
    def test_select_runs(self, db):
        rows = sqlenv.run_query(db, "SELECT COUNT(*) FROM products")
        assert rows == [(10,)]

    @pytest.mark.parametrize("evil", [
        "DELETE FROM products",
        "UPDATE products SET price = 0",
        "INSERT INTO products VALUES (99,'x','y',1.0,1)",
        "DROP TABLE products",
        "CREATE TABLE hack (x)",
        "ALTER TABLE products ADD COLUMN hacked TEXT",
        "SELECT 1; DROP TABLE products",              # stacked statements
        "ATTACH DATABASE '/tmp/x.db' AS x",
    ])
    def test_non_reads_denied_and_db_unchanged(self, db, evil):
        before = _count(db, "products")
        assert sqlenv.run_query(db, evil) is None
        assert _count(db, "products") == before  # nothing was written

    def test_garbage_returns_none(self, db):
        assert sqlenv.run_query(db, "not sql at all") is None
        assert sqlenv.run_query(db, None) is None


class TestParsing:
    def test_parse_sql_tagged(self):
        assert sqlenv.parse_sql("sure: <sql>SELECT 1;</sql> ok") == "SELECT 1"

    def test_parse_sql_bare_fallback(self):
        assert sqlenv.parse_sql("SELECT name FROM products").startswith("SELECT name")

    def test_parse_sql_none(self):
        assert sqlenv.parse_sql("I don't know") is None


class TestResultsMatch:
    def test_order_and_float_insensitive(self):
        assert sqlenv.results_match([(1,), (2,)], [(2,), (1,)])
        assert sqlenv.results_match([(12.5,)], [(12.5000001,)])
        assert not sqlenv.results_match([(1,)], [(2,)])
        assert not sqlenv.results_match(None, [(1,)])


class TestRewardShaper:
    def test_shaper_target_scores_full(self, db):
        reward, shaper = get_reward_fn("sqlite_tool"), get_shaper("sqlite_tool")
        for q, _ in sqlenv.COACH_QA + sqlenv.HELDOUT_QA:
            target = shaper(q, "some messy model output")
            assert reward(q, target) == 1.0, f"gold shaper failed for: {q}"

    def test_reward_grades(self, db):
        reward = get_reward_fn("sqlite_tool")
        q = sqlenv.HELDOUT_QA[0][0]  # "...'Toys' category?" -> COUNT
        assert reward(q, "no query here") == 0.0                       # no SQL
        assert reward(q, "<sql>SELECT * FROM nope</sql>") == 0.2       # errors
        assert reward(q, "<sql>SELECT COUNT(*) FROM customers</sql>") == 0.4  # runs, wrong
        assert reward(q, f"<sql>{sqlenv.gold_for(q)}</sql>") == 1.0    # correct

    def test_reward_zero_without_db(self):
        sqlenv.set_db(None)
        reward = get_reward_fn("sqlite_tool")
        q = sqlenv.HELDOUT_QA[0][0]
        assert reward(q, f"<sql>{sqlenv.gold_for(q)}</sql>") == 0.0
