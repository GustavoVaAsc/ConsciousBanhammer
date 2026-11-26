"""Disjoint-set forest with deterministic group ids, for duplicate groups and rule clusters."""

from collections import defaultdict
from collections.abc import Iterable, Mapping

from banhammer.core.errors import InvalidKeyError, UnknownKeyError


class UnionFind[KeyT: (str, int)]:
    """Incremental union-find over string or integer keys.

    Keys are plain `str` or plain `int` (one kind per instance, fixed by the first key) so that
    every group has a smallest member. Subclasses such as `bool` and `numpy.int64` are rejected.
    The group id of a group is that smallest key, which does not depend on the order or the
    direction of the unions that built the group: the same edges in any order give the same ids.
    `find` compresses paths and `union` links the smaller tree under the larger one.
    """

    def __init__(self, keys: Iterable[KeyT] = ()) -> None:
        """Start with `keys` as singleton groups."""
        self._parent: dict[KeyT, KeyT] = {}
        self._size: dict[KeyT, int] = {}
        self._smallest: dict[KeyT, KeyT] = {}
        self._kind: type | None = None
        for key in keys:
            self.add(key)

    def __len__(self) -> int:
        return len(self._parent)

    def __contains__(self, key: object) -> bool:
        return key in self._parent

    @property
    def group_count(self) -> int:
        """Number of groups."""
        return len(self._size)

    def add(self, key: KeyT) -> None:
        """Add `key` as a singleton group; a key already present is left alone.

        Raises:
            InvalidKeyError: `key` is not a plain `str` or `int`, or not of the kind of the
                keys already held. Nothing is changed.
        """
        self._validate(key)
        self._admit(key)

    def union(self, first: KeyT, second: KeyT) -> bool:
        """Merge the groups of the two keys, adding either key if it is new.

        Returns:
            Whether two different groups were merged.

        Raises:
            InvalidKeyError: A key is invalid as in `add`, or the two differ in kind. Nothing is
                changed.
        """
        self._validate(first, second)
        return self._merge(first, second)

    def union_all(self, pairs: Iterable[tuple[KeyT, KeyT]]) -> int:
        """Merge the groups of every pair, adding unseen keys.

        Returns:
            The number of merges that joined two different groups.

        Raises:
            InvalidKeyError: Some key is invalid as in `union`. Nothing is changed.
        """
        edges = list(pairs)
        self._validate(*(key for edge in edges for key in edge))
        return sum(self._merge(first, second) for first, second in edges)

    def find(self, key: KeyT) -> KeyT:
        """Representative of the group of `key`; it can change as groups merge.

        Raises:
            UnknownKeyError: `key` was never added.
        """
        if key not in self._parent:
            message = f"unknown key {key!r}"
            raise UnknownKeyError(message)
        root = key
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[key] != root:
            self._parent[key], key = root, self._parent[key]
        return root

    def connected(self, first: KeyT, second: KeyT) -> bool:
        """Whether both keys are in the same group.

        Raises:
            UnknownKeyError: A key was never added.
        """
        return self.find(first) == self.find(second)

    def group_id(self, key: KeyT) -> KeyT:
        """Smallest key of the group of `key`.

        Raises:
            UnknownKeyError: `key` was never added.
        """
        return self._smallest[self.find(key)]

    def group_size(self, key: KeyT) -> int:
        """Number of keys in the group of `key`.

        Raises:
            UnknownKeyError: `key` was never added.
        """
        return self._size[self.find(key)]

    def group_ids(self) -> Mapping[KeyT, KeyT]:
        """Group id of every key."""
        return {key: self.group_id(key) for key in self._parent}

    def groups(self) -> Mapping[KeyT, tuple[KeyT, ...]]:
        """Members of every group, sorted, by group id."""
        members: defaultdict[KeyT, list[KeyT]] = defaultdict(list)
        for key, group_id in self.group_ids().items():
            members[group_id].append(key)
        return {group_id: tuple(sorted(keys)) for group_id, keys in sorted(members.items())}

    def _validate(self, *keys: KeyT) -> None:
        kind = self._kind or (type(keys[0]) if keys else None)
        for key in keys:
            if type(key) is not kind or kind not in (str, int):
                message = f"keys must be plain str or int of one kind, got {key!r} ({type(key)})"
                raise InvalidKeyError(message)

    def _admit(self, key: KeyT) -> None:
        self._kind = type(key)
        if key not in self._parent:
            self._parent[key] = key
            self._size[key] = 1
            self._smallest[key] = key

    def _merge(self, first: KeyT, second: KeyT) -> bool:
        self._admit(first)
        self._admit(second)
        small, large = sorted((self.find(first), self.find(second)), key=self._size.__getitem__)
        if small == large:
            return False
        self._parent[small] = large
        self._size[large] += self._size.pop(small)
        self._smallest[large] = min(self._smallest.pop(small), self._smallest[large])
        return True
