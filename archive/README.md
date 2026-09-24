# Historical prototypes

`legacy/gnn_v1.py` and `legacy/create_feature.py` retain early research code. They are retained for historical inspection.

They are outside the supported pipeline. The GAT prototype uses global normalization before the train/test split; the feature draft has incomplete dictionary/array handling and duplicates bid inputs in its ask construction. These files must not be used to generate current reported results. Their supported replacements are `remoma.dataset.preprocessing` and the root training commands.

Historical versions and superseded documentation remain available in Git. Current behavior is defined by the package, tests, and [documentation](../docs/README.md).
