"""Real SQLite environment for the text→SQL tool-use demo.

The model learns to answer natural-language questions about a real database by
emitting a query:

    <sql>SELECT name FROM products ORDER BY price DESC LIMIT 1</sql>

We execute it against an actual SQLite engine and reward the model when the
result set matches a gold query's result. Because held-out questions use values
(categories, cities, names) that never appear in coaching, a correct answer can
only come from writing correct SQL — not from memorizing an answer.

Safety — the model's SQL is untrusted, so execution is jailed several ways:
  * connection opened read-only (`file:...?mode=ro`, `PRAGMA query_only=ON`);
  * a SQLite *authorizer* permits only read actions (SELECT / READ / FUNCTION)
    and denies every write, DDL, ATTACH, or pragma — the primary defense, which
    holds even if the text parser is fooled;
  * one statement per call (stacked queries raise), plus a progress-handler
    instruction budget so a pathological query can't spin.
No network, no filesystem writes, a throwaway DB file.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

# ---- schema + deterministic seed data ------------------------------------

_SCHEMA = """
CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, city TEXT);
CREATE TABLE products  (id INTEGER PRIMARY KEY, name TEXT, category TEXT,
                        price REAL, stock INTEGER);
CREATE TABLE orders    (id INTEGER PRIMARY KEY, customer_id INTEGER,
                        product_id INTEGER, quantity INTEGER, order_date TEXT);
"""

_CUSTOMERS = [
    (1, "Alice", "Seattle"), (2, "Bob", "Denver"), (3, "Carol", "Seattle"),
    (4, "Dan", "Austin"), (5, "Eve", "Denver"), (6, "Frank", "Boston"),
    (7, "Grace", "Austin"), (8, "Heidi", "Seattle"),
]
_PRODUCTS = [
    (1, "Notebook", "Books", 12.50, 120), (2, "Pen", "Office", 1.25, 500),
    (3, "Backpack", "Bags", 45.00, 30), (4, "Novel", "Books", 9.99, 75),
    (5, "Desk Lamp", "Office", 22.00, 40), (6, "Water Bottle", "Kitchen", 15.00, 60),
    (7, "Puzzle", "Toys", 8.00, 90), (8, "Action Figure", "Toys", 18.50, 25),
    (9, "Mug", "Kitchen", 7.50, 200), (10, "Headphones", "Electronics", 79.99, 15),
]
_ORDERS = [
    (1, 1, 1, 3, "2026-01-05"), (2, 1, 4, 1, "2026-01-06"), (3, 2, 3, 1, "2026-02-01"),
    (4, 3, 2, 10, "2026-02-03"), (5, 4, 10, 2, "2026-02-10"), (6, 5, 6, 4, "2026-03-01"),
    (7, 1, 7, 1, "2026-03-02"), (8, 6, 8, 2, "2026-03-05"), (9, 7, 9, 6, "2026-03-06"),
    (10, 8, 1, 2, "2026-03-08"), (11, 2, 5, 1, "2026-03-09"), (12, 3, 4, 5, "2026-03-10"),
]


def build_db(path: str) -> str:
    """(Re)create the demo database at `path`. Deterministic. Returns the path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        p.unlink()
    con = sqlite3.connect(str(p))
    try:
        con.executescript(_SCHEMA)
        con.executemany("INSERT INTO customers VALUES (?,?,?)", _CUSTOMERS)
        con.executemany("INSERT INTO products  VALUES (?,?,?,?,?)", _PRODUCTS)
        con.executemany("INSERT INTO orders    VALUES (?,?,?,?,?)", _ORDERS)
        con.commit()
    finally:
        con.close()
    return str(p)


SCHEMA_DESCRIPTION = (
    "customers(id, name, city), "
    "products(id, name, category, price, stock), "
    "orders(id, customer_id, product_id, quantity, order_date)"
)

# ---- safe read-only execution --------------------------------------------

_OK = getattr(sqlite3, "SQLITE_OK", 0)
_DENY = getattr(sqlite3, "SQLITE_DENY", 1)
_ALLOWED_ACTIONS = {
    getattr(sqlite3, "SQLITE_SELECT", 21),
    getattr(sqlite3, "SQLITE_READ", 20),
    getattr(sqlite3, "SQLITE_FUNCTION", 31),
    getattr(sqlite3, "SQLITE_RECURSIVE", 33),
}
_PROGRESS_BUDGET = 100_000  # abort a query that executes more than this many VM ops

_SQL_RE = re.compile(r"<sql>\s*(?P<q>.+?)\s*</sql>", re.IGNORECASE | re.DOTALL)


def parse_sql(text: str) -> str | None:
    """Return the query inside the first <sql>…</sql> block, or None.

    Falls back to a bare leading SELECT/WITH statement if the tags are absent.
    """
    m = _SQL_RE.search(text or "")
    q = m.group("q").strip() if m else None
    if q is None:
        m2 = re.search(r"\b(select|with)\b.+", text or "", re.IGNORECASE | re.DOTALL)
        q = m2.group(0).strip() if m2 else None
    if not q:
        return None
    return q.rstrip(";").strip()  # drop a trailing ; so it stays one statement


def _authorizer(action, arg1, arg2, dbname, trigger):
    return _OK if action in _ALLOWED_ACTIONS else _DENY


def run_query(db_path: str, sql: str | None):
    """Execute a read-only query; return list-of-row-tuples, or None on any error.

    None covers: no query, denied action (write/DDL/attach/pragma), syntax error,
    multiple statements, or exceeding the instruction budget.
    """
    if not sql:
        return None
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error:
        return None
    try:
        con.execute("PRAGMA query_only = ON")
        con.set_authorizer(_authorizer)
        budget = [_PROGRESS_BUDGET]

        def _progress():
            budget[0] -= 1
            return 1 if budget[0] <= 0 else 0  # non-zero aborts the query

        con.set_progress_handler(_progress, 1000)
        cur = con.execute(sql)  # raises on >1 statement
        return [tuple(r) for r in cur.fetchall()]
    except (sqlite3.Error, ValueError):
        return None
    finally:
        con.set_authorizer(None)
        con.close()


def _norm_rows(rows) -> list | None:
    """Order-insensitive, type-loose normal form for comparing result sets."""
    if rows is None:
        return None
    out = []
    for r in rows:
        cells = []
        for c in r:
            if isinstance(c, float):
                cells.append(f"{c:.4f}")
            else:
                cells.append(str(c))
        out.append(tuple(cells))
    return sorted(out)


def results_match(a, b) -> bool:
    """True iff two result sets are equal ignoring row order and numeric formatting."""
    na, nb = _norm_rows(a), _norm_rows(b)
    return na is not None and nb is not None and na == nb


# ---- questions (coach + held-out) with gold SQL --------------------------
# Held-out deliberately uses categories/cities/products that NEVER appear in the
# coach set, so a correct executed answer proves the model wrote correct SQL
# rather than memorizing a value.

COACH_QA = [
    ("How many products are in the 'Books' category?",
     "SELECT COUNT(*) FROM products WHERE category = 'Books'"),
    ("What is the total stock across all products?",
     "SELECT SUM(stock) FROM products"),
    ("List the names of customers from Seattle.",
     "SELECT name FROM customers WHERE city = 'Seattle'"),
    ("What is the name of the most expensive product?",
     "SELECT name FROM products ORDER BY price DESC LIMIT 1"),
    ("How many customers are there?",
     "SELECT COUNT(*) FROM customers"),
    ("What is the average price of products in the 'Office' category?",
     "SELECT AVG(price) FROM products WHERE category = 'Office'"),
    ("Which product has the lowest stock? Give its name.",
     "SELECT name FROM products ORDER BY stock ASC LIMIT 1"),
    ("How many orders were placed by the customer named 'Alice'?",
     "SELECT COUNT(*) FROM orders JOIN customers ON orders.customer_id = customers.id "
     "WHERE customers.name = 'Alice'"),
]

HELDOUT_QA = [
    ("How many products are in the 'Toys' category?",
     "SELECT COUNT(*) FROM products WHERE category = 'Toys'"),
    ("List the names of customers from Denver.",
     "SELECT name FROM customers WHERE city = 'Denver'"),
    ("What is the name of the cheapest product?",
     "SELECT name FROM products ORDER BY price ASC LIMIT 1"),
    ("What is the total stock of products in the 'Kitchen' category?",
     "SELECT SUM(stock) FROM products WHERE category = 'Kitchen'"),
    ("Which product has the highest stock? Give its name.",
     "SELECT name FROM products ORDER BY stock DESC LIMIT 1"),
    ("How many orders were placed by the customer named 'Carol'?",
     "SELECT COUNT(*) FROM orders JOIN customers ON orders.customer_id = customers.id "
     "WHERE customers.name = 'Carol'"),
]

_GOLD = {q: g for q, g in COACH_QA + HELDOUT_QA}


def gold_for(question: str) -> str | None:
    return _GOLD.get(question.strip())


# ---- module-level DB path (set by the demo, read by the reward) ----------

_DB_PATH: str | None = None


def set_db(path: str) -> None:
    global _DB_PATH
    _DB_PATH = path


def db_path() -> str | None:
    return _DB_PATH
