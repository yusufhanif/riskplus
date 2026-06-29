from __future__ import annotations


def test_key_modules_importable() -> None:
    import riskplus_core.analytics  # noqa: F401
    import riskplus_core.attribution  # noqa: F401
    import riskplus_core.baseline  # noqa: F401
    import riskplus_core.contribution  # noqa: F401
    import riskplus_core.data  # noqa: F401
    import riskplus_core.interfaces  # noqa: F401
    import riskplus_core.engine  # noqa: F401
    import riskplus_core.factors  # noqa: F401
    import riskplus_core.models  # noqa: F401
    import riskplus_core.portfolio  # noqa: F401
    import riskplus_core.explanations  # noqa: F401
    import riskplus_core.quality  # noqa: F401
    import riskplus_core.reporting  # noqa: F401
    import riskplus_core.risk  # noqa: F401
    import riskplus_core.simulation  # noqa: F401
    import riskplus_ui.report_tabs  # noqa: F401
    import riskplus_ui.workflow  # noqa: F401