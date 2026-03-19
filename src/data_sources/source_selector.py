from data_sources.aemet_source import AemetWeatherSource


def get_data_source_instance(source_name: str, config: dict):
    """
    Get a data source instance based on name.

    Args:
        source_name: Name of the data source (e.g. 'AEMET')
        config: Configuration dictionary

    Returns:
        Data source instance, or None if not found
    """
    source_config = next(
        (source for source in config.get('data_sources', []) if source['name'] == source_name),
        None
    )

    if not source_config:
        return None

    if source_name == 'AEMET':
        return AemetWeatherSource(source_config)

    return None
