from financials.main import main


def test_main_runs(capsys):
    main()
    captured = capsys.readouterr()
    assert "Hello from financials!" in captured.out
