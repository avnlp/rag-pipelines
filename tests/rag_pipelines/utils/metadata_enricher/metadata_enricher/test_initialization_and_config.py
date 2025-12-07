"""Tests for initialization and configuration of MetadataEnricher.

Verifies that the enricher can be instantiated with different configurations
(enrichment mode, batch size), that default values are sensible, that the
EnrichmentMode enum has correct string values, and that configuration is
correctly stored and accessible on enricher instances.
"""

from rag_pipelines.utils.metadata_enricher import (
    EnrichmentConfig,
    EnrichmentMode,
    MetadataEnricher,
)


class TestMetadataEnricherInitialization:
    """Test initialization and configuration of MetadataEnricher.

    Validates EnrichmentMode enum values, EnrichmentConfig default and custom
    values, and that configuration is correctly stored and accessible on
    enricher instances.
    """

    def test_enrichment_mode_enum(self) -> None:
        """Verify EnrichmentMode enum has correct string values.

        Ensures the three enrichment modes (MINIMAL, DYNAMIC, FULL) map to
        their expected lowercase string representations.
        """
        assert EnrichmentMode.MINIMAL.value == "minimal"
        assert EnrichmentMode.DYNAMIC.value == "dynamic"
        assert EnrichmentMode.FULL.value == "full"

    def test_enrichment_config_defaults(self) -> None:
        """Verify EnrichmentConfig uses correct default values.

        Ensures that when EnrichmentConfig is instantiated without arguments,
        it defaults to FULL enrichment mode and batch size of 10.
        """
        config = EnrichmentConfig()
        assert config.mode == EnrichmentMode.FULL
        assert config.batch_size == 10

    def test_enrichment_config_custom(self) -> None:
        """Verify EnrichmentConfig accepts and stores custom values.

        Ensures that EnrichmentConfig can be instantiated with custom mode
        and batch_size, and that the provided values are stored correctly.
        """
        config = EnrichmentConfig(mode=EnrichmentMode.MINIMAL, batch_size=5)
        assert config.mode == EnrichmentMode.MINIMAL
        assert config.batch_size == 5

    def test_metadata_enricher_config_immutable(self) -> None:
        """Verify MetadataEnricher stores and exposes configuration correctly.

        Ensures that configuration passed to MetadataEnricher is accessible
        via the public .config property with all values preserved.
        """
        config = EnrichmentConfig(mode=EnrichmentMode.MINIMAL, batch_size=20)
        enricher = MetadataEnricher(config)

        assert enricher.config.mode == EnrichmentMode.MINIMAL
        assert enricher.config.batch_size == 20

    def test_enricher_default_constructor(self) -> None:
        """Verify MetadataEnricher default constructor uses FULL mode.

        Ensures that MetadataEnricher() with no arguments defaults to FULL
        enrichment mode via the default EnrichmentConfig.
        """
        enricher = MetadataEnricher()
        assert enricher.config.mode == EnrichmentMode.FULL

    def test_enricher_with_custom_mode(self) -> None:
        """Verify MetadataEnricher accepts and stores custom configuration.

        Ensures that MetadataEnricher can be instantiated with a custom
        EnrichmentConfig and that the mode is accessible via .config.
        """
        config = EnrichmentConfig(mode=EnrichmentMode.MINIMAL)
        enricher = MetadataEnricher(config)
        assert enricher.config.mode == EnrichmentMode.MINIMAL
