class ConfigRegistryError(Exception):
    pass


class ConfigNotFoundError(ConfigRegistryError):
    pass


class DuplicateConfigError(ConfigRegistryError):
    pass


class SchemaNotInitializedError(ConfigRegistryError):
    pass
