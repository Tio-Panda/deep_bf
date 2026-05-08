from .utils import get_names_groups


class GlobalSamplesIdx:
    def __init__(self, query):
        self.groups = get_names_groups(query)

        id = 0
        name2id = {}

        for _, mini_df in self.groups:
            group_names = mini_df["name"]

            for name in group_names:
                name2id[name] = id

            id += 1

        self.name2id = name2id

    def __getitem__(self, key):
        return self.name2id[key]
