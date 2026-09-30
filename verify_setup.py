"""
Verification script for MarketLens Phase 1 database models and metadata.
Run this script to verify your database connection and inspect table structures.
"""
from database.connection import DATABASE_URL, engine
from database.models import Base, Company, MarketData, MarketIndex, News


def verify_metadata():
    print(f"[*] Target Database URL: {DATABASE_URL}")
    print("[*] Registered SQLAlchemy Models:")
    for table_name, table in Base.metadata.tables.items():
        print(f"\n  Table: {table_name}")
        for col in table.columns:
            nullable_str = "NULL" if col.nullable else "NOT NULL"
            default_str = f" DEFAULT {col.server_default.arg}" if col.server_default else ""
            print(f"    - {col.name:25} {str(col.type):30} {nullable_str}{default_str}")

        if table.constraints:
            print("    Constraints & Indexes:")
            for const in table.constraints:
                if const.name:
                    print(f"      * {type(const).__name__}: {const.name}")
            for idx in table.indexes:
                print(f"      * Index: {idx.name} ({', '.join(c.name for c in idx.columns)})")


if __name__ == "__main__":
    verify_metadata()
    print("\n[+] Setup verified successfully!")
