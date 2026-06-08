from fairaudit.audit import emit, verify


def test_chain_appends_and_verifies(tmp_path):
    ledger = tmp_path / "l.ndjson"
    emit("fairness_pre", "job", {"summary": {"demographic_parity_diff": 0.2}}, ledger_path=ledger)
    emit("fairness_post", "job", {"summary": {"demographic_parity_diff": 0.0}}, ledger_path=ledger)
    ok, n = verify(ledger)
    assert ok and n == 2


def test_tamper_breaks_chain(tmp_path):
    ledger = tmp_path / "l.ndjson"
    emit("fairness_pre", "job", {"a": 1}, ledger_path=ledger)
    emit("fairness_post", "job", {"a": 2}, ledger_path=ledger)
    lines = ledger.read_text().splitlines()
    lines[0] = lines[0].replace('"a":1', '"a":999')
    ledger.write_text("\n".join(lines) + "\n")
    ok, _ = verify(ledger)
    assert ok is False


def test_empty_ledger_ok(tmp_path):
    ok, n = verify(tmp_path / "missing.ndjson")
    assert ok and n == 0
