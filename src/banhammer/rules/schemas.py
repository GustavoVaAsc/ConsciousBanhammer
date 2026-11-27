"""Pandera schemas of the S1 artifacts: the rules and canonical-clusters boundaries (D28).

`RulesetsSchema`, `CandidatePairsSchema` and `SaturationCurveSchema` are checked the same way
but are supporting artifacts, not boundaries. Each schema is strict: no extra columns, no
coercion, column order fixed.
"""

import datetime

import pandera.polars as pa
import polars as pl

from banhammer.core.constants import Language
from banhammer.rules.constants import (
    CLUSTER_ID_PATTERN,
    CLUSTER_ID_PREFIX,
    COSINE_DISTANCE_MAX,
    COSINE_MAX,
    COSINE_MIN,
    FINAL_RESOLUTIONS,
    PLATFORM_PATTERN,
    PROBABILITY_MAX,
    PROBABILITY_MIN,
    RULE_ID_PATTERN,
    RULESET_ID_PATTERN,
    SOURCE_URL_PATTERN,
    PairOutcome,
    RuleType,
)

NON_EMPTY = {"min_value": 1}


class RulesSchema(pa.DataFrameModel):
    """One row per atomic rule: the eight fields of proposal 4.1."""

    rule_id: str = pa.Field(unique=True, str_matches=RULE_ID_PATTERN)
    ruleset_id: str = pa.Field(str_matches=RULESET_ID_PATTERN)
    text: str = pa.Field(str_length=NON_EMPTY)
    scope: str | None = pa.Field(nullable=True, str_length=NON_EMPTY)
    exceptions: str | None = pa.Field(nullable=True, str_length=NON_EMPTY)
    sanction_hint: str | None = pa.Field(nullable=True, str_length=NON_EMPTY)
    rule_type: str = pa.Field(isin=[member.value for member in RuleType])
    discretion_flag: bool

    class Config:
        """Strict columns in a fixed order, no coercion."""

        strict = True
        ordered = True


class RulesetsSchema(pa.DataFrameModel):
    """One row per ruleset: stratification attributes, provenance and size."""

    ruleset_id: str = pa.Field(unique=True, str_matches=RULESET_ID_PATTERN)
    platform: str = pa.Field(str_matches=PLATFORM_PATTERN)
    language: str = pa.Field(isin=[member.value for member in Language])
    source_url: str = pa.Field(str_matches=SOURCE_URL_PATTERN)
    harvest_date: datetime.date
    terms_note: str = pa.Field(str_length=NON_EMPTY)
    rule_count: int = pa.Field(ge=1)

    class Config:
        """Strict columns in a fixed order, no coercion."""

        strict = True
        ordered = True


class CandidatePairsSchema(pa.DataFrameModel):
    """One row per unordered candidate pair, `rule_a < rule_b`, with its scores and outcome."""

    rule_a: str = pa.Field(str_matches=RULE_ID_PATTERN)
    rule_b: str = pa.Field(str_matches=RULE_ID_PATTERN)
    cosine: float = pa.Field(ge=COSINE_MIN, le=COSINE_MAX)
    p_entail_ab: float = pa.Field(ge=PROBABILITY_MIN, le=PROBABILITY_MAX)
    p_entail_ba: float = pa.Field(ge=PROBABILITY_MIN, le=PROBABILITY_MAX)
    outcome: str = pa.Field(isin=[member.value for member in PairOutcome])

    class Config:
        """Strict columns in a fixed order, unique pairs."""

        strict = True
        ordered = True
        unique = ("rule_a", "rule_b")

    @pa.dataframe_check
    @classmethod
    def pair_is_ordered(cls, data: pa.PolarsData) -> pl.LazyFrame:
        """Each pair is stored once, with the smaller rule id first."""
        return data.lazyframe.select(pl.col("rule_a") < pl.col("rule_b"))


class CanonicalClustersSchema(pa.DataFrameModel):
    """One row per rule: the canonical cluster it belongs to and the cluster's checks.

    The cluster id is `CLUSTER_ID_PREFIX` plus the smallest rule id of the cluster, so it does
    not depend on row order or on the order the edges were found in.
    """

    cluster_id: str = pa.Field(str_matches=CLUSTER_ID_PATTERN)
    rule_id: str = pa.Field(unique=True, str_matches=RULE_ID_PATTERN)
    ruleset_id: str = pa.Field(str_matches=RULESET_ID_PATTERN)
    cluster_size: int = pa.Field(ge=1)
    diameter: float = pa.Field(ge=0, le=COSINE_DISTANCE_MAX)
    resolution: str = pa.Field(isin=[member.value for member in FINAL_RESOLUTIONS])

    class Config:
        """Strict columns in a fixed order, no coercion."""

        strict = True
        ordered = True

    @pa.dataframe_check
    @classmethod
    def cluster_id_is_prefix_plus_smallest_rule(cls, data: pa.PolarsData) -> pl.LazyFrame:
        """The cluster id names the smallest member."""
        expected = pl.lit(CLUSTER_ID_PREFIX) + pl.col("rule_id").min().over("cluster_id")
        return data.lazyframe.select(pl.col("cluster_id") == expected)

    @pa.dataframe_check
    @classmethod
    def cluster_size_counts_members(cls, data: pa.PolarsData) -> pl.LazyFrame:
        """The size column equals the number of rows of the cluster."""
        return data.lazyframe.select(pl.col("cluster_size") == pl.len().over("cluster_id"))

    @pa.dataframe_check
    @classmethod
    def cluster_attributes_are_shared(cls, data: pa.PolarsData) -> pl.LazyFrame:
        """Diameter and resolution are one value per cluster."""
        shared = (pl.col("diameter").n_unique().over("cluster_id") == 1) & (
            pl.col("resolution").n_unique().over("cluster_id") == 1
        )
        return data.lazyframe.select(shared)


class SaturationCurveSchema(pa.DataFrameModel):
    """Cumulative canonical clusters after each step of rulesets in harvest order."""

    rulesets_added: int = pa.Field(unique=True, ge=1)
    rule_count: int = pa.Field(ge=1)
    cluster_count: int = pa.Field(ge=1)
    new_clusters: int = pa.Field(ge=0)

    class Config:
        """Strict columns in a fixed order, no coercion."""

        strict = True
        ordered = True

    @pa.dataframe_check
    @classmethod
    def rulesets_increase(cls, data: pa.PolarsData) -> pl.LazyFrame:
        """Points follow one another in harvest order."""
        return data.lazyframe.select(pl.col("rulesets_added").diff().fill_null(1) > 0)

    @pa.dataframe_check
    @classmethod
    def new_clusters_are_the_increase(cls, data: pa.PolarsData) -> pl.LazyFrame:
        """New clusters are the growth of the cluster count since the previous point."""
        growth = pl.col("cluster_count") - pl.col("cluster_count").shift(1).fill_null(0)
        return data.lazyframe.select(pl.col("new_clusters") == growth)
