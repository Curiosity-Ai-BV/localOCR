from core.json_extract import (
    extract_structured_data,
)


def test_extract_from_fenced_block():
    text = """
    Some intro...
    ```json
    {"invoice": "123", "amount": 42.5}
    ```
    trailing
    """
    obj = extract_structured_data(text, ["invoice", "amount"])  # type: ignore
    assert obj.get("invoice") == "123"
    assert obj.get("amount") == 42.5


def test_extract_by_brace_scan():
    text = "prefix {\n  \"company\": \"Acme\", \n  \"total\": 99\n} suffix"
    obj = extract_structured_data(text, ["company"])  # type: ignore
    assert obj.get("company") == "Acme"
    assert obj.get("total") == 99


def test_extract_heuristics():
    text = "Invoice number: 001-A; Date: 2024-01-01; Total amount = $12.00"
    obj = extract_structured_data(text, ["Invoice number", "Date", "Total amount"])  # type: ignore
    assert obj.get("Invoice number") == "001-A"
    assert obj.get("Date") == "2024-01-01"
    assert "$12.00" in str(obj.get("Total amount"))

