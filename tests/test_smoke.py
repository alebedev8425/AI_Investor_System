def test_repo_imports():
    import system  # package root
    import system.Controller.run_manager as rm
    assert hasattr(rm, "RunManager")