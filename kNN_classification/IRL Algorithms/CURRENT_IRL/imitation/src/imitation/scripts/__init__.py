"""Command-line scripts."""

# Add our custom environments to Gym registry.

try:
    # pytype: disable=import-error
    import seals_dir  # noqa: F401

    # pytype: enable=import-error
except ImportError:
    pass
