"""
Clustering and Deduplication Service for AI Forecasting Pipeline.

Provides event clustering using multiple algorithms (Agglomerative, K-Means, DBSCAN)
with semantic similarity, canonical event selection, and citation merging.
"""

import json
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

import numpy as np
import structlog
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.metrics import silhouette_score

from services.embedding import Embedding, compute_distance_matrix
from db.models import Event, EventCluster as EventClusterModel

logger = structlog.get_logger(__name__)


class ClusteringAlgorithm(str, Enum):
    """Supported clustering algorithms."""
    AGGLOMERATIVE = "agglomerative"
    KMEANS = "kmeans"
    DBSCAN = "dbscan"


class ClusteringError(Exception):
    """Base exception for clustering-related errors."""
    pass


@dataclass
class EventCluster:
    """
    Represents a cluster of similar events.

    Attributes:
        cluster_id: Cluster identifier
        centroid_event_id: ID of the canonical/centroid event
        member_event_ids: List of all event IDs in cluster
        label: Human-readable label for the cluster
        confidence_score: Average confidence of events in cluster
        merged_citations: List of all source citations
        merged_actors: List of all unique actors
    """
    cluster_id: int
    centroid_event_id: int
    member_event_ids: List[int]
    label: Optional[str] = None
    confidence_score: float = 0.0
    merged_citations: List[Dict[str, str]] = None
    merged_actors: List[str] = None

    def __post_init__(self):
        if self.merged_citations is None:
            self.merged_citations = []
        if self.merged_actors is None:
            self.merged_actors = []

    def to_model(self, run_id: int) -> EventClusterModel:
        """
        Convert to database model.

        Args:
            run_id: ID of the pipeline run

        Returns:
            EventClusterModel instance
        """
        return EventClusterModel(
            run_id=run_id,
            label=self.label,
            centroid_event_id=self.centroid_event_id,
            member_ids=json.dumps(self.member_event_ids),
        )


@dataclass
class ClusteringResult:
    """
    Results from clustering operation.

    Attributes:
        clusters: List of EventCluster objects
        algorithm: Algorithm used
        n_clusters: Number of clusters found
        silhouette_score: Quality metric (-1 to 1, higher is better)
        cluster_size_distribution: Dict mapping cluster size to count
        original_count: Number of events before clustering
        deduplicated_count: Number of canonical events after clustering
    """
    clusters: List[EventCluster]
    algorithm: ClusteringAlgorithm
    n_clusters: int
    silhouette_score: float
    cluster_size_distribution: Dict[int, int]
    original_count: int
    deduplicated_count: int

    def get_deduplication_rate(self) -> float:
        """Calculate deduplication rate as percentage."""
        if self.original_count == 0:
            return 0.0
        return (1.0 - self.deduplicated_count / self.original_count) * 100


class ClusteringService:
    """
    Service for clustering events and deduplication.

    Supports multiple clustering algorithms with configurable parameters
    and automatic canonical event selection.
    """

    def __init__(
        self,
        algorithm: ClusteringAlgorithm = ClusteringAlgorithm.AGGLOMERATIVE,
        distance_threshold: float = 0.2,
        min_cluster_size: int = 1,
    ):
        """
        Initialize clustering service.

        Args:
            algorithm: Clustering algorithm to use
            distance_threshold: Distance threshold for clustering (cosine distance)
            min_cluster_size: Minimum events per cluster
        """
        self.algorithm = algorithm
        self.distance_threshold = distance_threshold
        self.min_cluster_size = min_cluster_size
        self.logger = structlog.get_logger(__name__)

        self.logger.info(
            "clustering_service_initialized",
            algorithm=algorithm,
            distance_threshold=distance_threshold,
            min_cluster_size=min_cluster_size,
        )

    def _select_centroid_event(
        self,
        cluster_member_indices: List[int],
        events: List[Event],
        embeddings: List[Embedding],
    ) -> int:
        """
        Select the canonical/centroid event for a cluster.

        Selection criteria:
        1. Highest confidence score
        2. If tied, most central (closest to cluster center in embedding space)
        3. If tied, longest description

        Args:
            cluster_member_indices: Indices of events in this cluster
            events: List of all events
            embeddings: List of all embeddings

        Returns:
            Index of the centroid event
        """
        if len(cluster_member_indices) == 1:
            return cluster_member_indices[0]

        # Get member events and embeddings
        member_events = [events[i] for i in cluster_member_indices]
        member_embeddings = [embeddings[i].vector for i in cluster_member_indices]

        # Compute cluster center
        center = np.mean(member_embeddings, axis=0)

        # Score each event
        best_idx = None
        best_score = -float('inf')

        for i, idx in enumerate(cluster_member_indices):
            event = events[idx]

            # Primary: confidence score
            confidence = event.confidence or 0.5

            # Secondary: centrality (negative distance to center)
            distance_to_center = np.linalg.norm(member_embeddings[i] - center)
            centrality = -distance_to_center

            # Tertiary: content length
            body_length = len(event.body or "")

            # Combined score (weighted)
            score = (
                confidence * 10.0 +          # Confidence is most important
                centrality * 1.0 +            # Centrality matters
                body_length * 0.001           # Length is tiebreaker
            )

            if score > best_score:
                best_score = score
                best_idx = idx

        return best_idx

    def _merge_citations(
        self,
        cluster_member_indices: List[int],
        events: List[Event],
        documents: Optional[Dict[int, Any]] = None,
    ) -> List[Dict[str, str]]:
        """
        Merge citations from all events in a cluster.

        Args:
            cluster_member_indices: Indices of events in cluster
            events: List of all events
            documents: Optional dict mapping document_id to document data

        Returns:
            List of citation dicts with url and quote
        """
        citations = []
        seen_urls = set()

        for idx in cluster_member_indices:
            event = events[idx]

            # Get URL from document if available
            url = None
            if documents and event.document_id in documents:
                url = documents[event.document_id].get('url')

            # Skip if no URL or duplicate
            if not url or url in seen_urls:
                continue

            seen_urls.add(url)

            # Add citation
            citations.append({
                'url': url,
                'quote': event.headline or event.body or "",
                'event_id': event.id,
            })

        return citations

    def _merge_actors(
        self,
        cluster_member_indices: List[int],
        events: List[Event],
    ) -> List[str]:
        """
        Merge and deduplicate actors from all events in a cluster.

        Args:
            cluster_member_indices: Indices of events in cluster
            events: List of all events

        Returns:
            List of unique actors
        """
        all_actors = set()

        for idx in cluster_member_indices:
            event = events[idx]
            if event.actors:
                try:
                    actors = json.loads(event.actors)
                    if isinstance(actors, list):
                        all_actors.update(actors)
                except json.JSONDecodeError:
                    # Try splitting by comma
                    actors = [a.strip() for a in event.actors.split(',')]
                    all_actors.update(actors)

        return sorted(list(all_actors))

    def _compute_cluster_label(
        self,
        cluster_member_indices: List[int],
        events: List[Event],
        centroid_idx: int,
    ) -> str:
        """
        Generate a human-readable label for a cluster.

        Args:
            cluster_member_indices: Indices of events in cluster
            events: List of all events
            centroid_idx: Index of centroid event

        Returns:
            Label string
        """
        centroid_event = events[centroid_idx]

        # Use headline of centroid event
        if centroid_event.headline:
            label = centroid_event.headline[:80]
            if len(centroid_event.headline) > 80:
                label += "..."
            return label

        # Fallback to body
        if centroid_event.body:
            label = centroid_event.body[:80]
            if len(centroid_event.body) > 80:
                label += "..."
            return label

        return f"Cluster with {len(cluster_member_indices)} events"

    def cluster_events(
        self,
        events: List[Event],
        embeddings: List[Embedding],
        documents: Optional[Dict[int, Any]] = None,
        n_clusters: Optional[int] = None,
    ) -> ClusteringResult:
        """
        Cluster events based on embedding similarity.

        Args:
            events: List of Event objects
            embeddings: List of Embedding objects (must match events order)
            documents: Optional dict mapping document_id to document data
            n_clusters: Number of clusters (for K-Means only)

        Returns:
            ClusteringResult with clusters and diagnostics

        Raises:
            ClusteringError: If clustering fails
        """
        if len(events) != len(embeddings):
            raise ClusteringError(
                f"Events and embeddings length mismatch: {len(events)} vs {len(embeddings)}"
            )

        if len(events) == 0:
            self.logger.warning("cluster_events_empty_input")
            return ClusteringResult(
                clusters=[],
                algorithm=self.algorithm,
                n_clusters=0,
                silhouette_score=0.0,
                cluster_size_distribution={},
                original_count=0,
                deduplicated_count=0,
            )

        self.logger.info(
            "clustering_start",
            algorithm=self.algorithm,
            n_events=len(events),
            n_clusters=n_clusters,
        )

        # Extract vectors
        vectors = np.array([emb.vector for emb in embeddings])

        # Perform clustering based on algorithm
        try:
            if self.algorithm == ClusteringAlgorithm.AGGLOMERATIVE:
                labels = self._cluster_agglomerative(vectors)
            elif self.algorithm == ClusteringAlgorithm.KMEANS:
                if n_clusters is None:
                    # Estimate reasonable number of clusters
                    n_clusters = max(1, len(events) // 5)
                labels = self._cluster_kmeans(vectors, n_clusters)
            elif self.algorithm == ClusteringAlgorithm.DBSCAN:
                labels = self._cluster_dbscan(vectors)
            else:
                raise ClusteringError(f"Unsupported algorithm: {self.algorithm}")

        except Exception as e:
            self.logger.error("clustering_failed", error=str(e), error_type=type(e).__name__)
            raise ClusteringError(f"Clustering failed: {e}")

        # Build clusters
        clusters = self._build_clusters(labels, events, embeddings, documents)

        # Compute diagnostics
        try:
            if len(set(labels)) > 1:  # Need at least 2 clusters for silhouette score
                sil_score = silhouette_score(vectors, labels, metric='cosine')
            else:
                sil_score = 0.0
        except Exception:
            sil_score = 0.0

        # Cluster size distribution
        size_dist: Dict[int, int] = {}
        for cluster in clusters:
            size = len(cluster.member_event_ids)
            size_dist[size] = size_dist.get(size, 0) + 1

        result = ClusteringResult(
            clusters=clusters,
            algorithm=self.algorithm,
            n_clusters=len(clusters),
            silhouette_score=sil_score,
            cluster_size_distribution=size_dist,
            original_count=len(events),
            deduplicated_count=len(clusters),
        )

        self.logger.info(
            "clustering_complete",
            n_clusters=result.n_clusters,
            silhouette_score=result.silhouette_score,
            deduplication_rate=result.get_deduplication_rate(),
            size_distribution=size_dist,
        )

        return result

    def _cluster_agglomerative(self, vectors: np.ndarray) -> np.ndarray:
        """
        Perform agglomerative clustering with cosine distance.

        Args:
            vectors: Array of embedding vectors (n_samples x n_features)

        Returns:
            Array of cluster labels
        """
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=self.distance_threshold,
            metric='cosine',
            linkage='average',
        )

        labels = clusterer.fit_predict(vectors)

        self.logger.debug(
            "agglomerative_clustering_complete",
            n_clusters=len(set(labels)),
            n_samples=len(vectors),
        )

        return labels

    def _cluster_kmeans(self, vectors: np.ndarray, n_clusters: int) -> np.ndarray:
        """
        Perform K-Means clustering.

        Args:
            vectors: Array of embedding vectors
            n_clusters: Number of clusters

        Returns:
            Array of cluster labels
        """
        # Normalize vectors for cosine distance
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normalized_vectors = vectors / (norms + 1e-10)

        clusterer = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
        )

        labels = clusterer.fit_predict(normalized_vectors)

        self.logger.debug(
            "kmeans_clustering_complete",
            n_clusters=n_clusters,
            inertia=clusterer.inertia_,
        )

        return labels

    def _cluster_dbscan(self, vectors: np.ndarray) -> np.ndarray:
        """
        Perform DBSCAN clustering.

        Args:
            vectors: Array of embedding vectors

        Returns:
            Array of cluster labels
        """
        # DBSCAN uses eps parameter (max distance for neighborhood)
        # Convert our distance_threshold to eps
        eps = self.distance_threshold

        clusterer = DBSCAN(
            eps=eps,
            min_samples=self.min_cluster_size,
            metric='cosine',
        )

        labels = clusterer.fit_predict(vectors)

        # DBSCAN uses -1 for noise points; reassign each noise point to its own cluster
        max_label = labels.max()
        noise_points = np.where(labels == -1)[0]

        for i, noise_idx in enumerate(noise_points):
            labels[noise_idx] = max_label + 1 + i

        self.logger.debug(
            "dbscan_clustering_complete",
            n_clusters=len(set(labels)),
            n_noise_points=len(noise_points),
        )

        return labels

    def _build_clusters(
        self,
        labels: np.ndarray,
        events: List[Event],
        embeddings: List[Embedding],
        documents: Optional[Dict[int, Any]],
    ) -> List[EventCluster]:
        """
        Build EventCluster objects from cluster labels.

        Args:
            labels: Array of cluster labels
            events: List of events
            embeddings: List of embeddings
            documents: Optional document data

        Returns:
            List of EventCluster objects
        """
        # Group indices by cluster label
        cluster_map: Dict[int, List[int]] = {}
        for idx, label in enumerate(labels):
            if label not in cluster_map:
                cluster_map[label] = []
            cluster_map[label].append(idx)

        clusters = []

        for cluster_id, member_indices in cluster_map.items():
            # Select centroid event
            centroid_idx = self._select_centroid_event(member_indices, events, embeddings)
            centroid_event = events[centroid_idx]

            # Get event IDs (assuming events have id field)
            member_event_ids = [events[i].id for i in member_indices]
            centroid_event_id = centroid_event.id

            # Compute average confidence
            confidences = [events[i].confidence or 0.5 for i in member_indices]
            avg_confidence = np.mean(confidences)

            # Merge citations and actors
            citations = self._merge_citations(member_indices, events, documents)
            actors = self._merge_actors(member_indices, events)

            # Generate label
            label = self._compute_cluster_label(member_indices, events, centroid_idx)

            cluster = EventCluster(
                cluster_id=cluster_id,
                centroid_event_id=centroid_event_id,
                member_event_ids=member_event_ids,
                label=label,
                confidence_score=float(avg_confidence),
                merged_citations=citations,
                merged_actors=actors,
            )

            clusters.append(cluster)

        # Sort by cluster size (descending)
        clusters.sort(key=lambda c: len(c.member_event_ids), reverse=True)

        return clusters


# Convenience functions

def cluster_events(
    events: List[Event],
    embeddings: List[Embedding],
    algorithm: ClusteringAlgorithm = ClusteringAlgorithm.AGGLOMERATIVE,
    distance_threshold: float = 0.2,
    n_clusters: Optional[int] = None,
    documents: Optional[Dict[int, Any]] = None,
) -> ClusteringResult:
    """
    Convenience function to cluster events.

    Args:
        events: List of Event objects
        embeddings: List of Embedding objects
        algorithm: Clustering algorithm to use
        distance_threshold: Distance threshold for clustering
        n_clusters: Number of clusters (for K-Means)
        documents: Optional document data for citations

    Returns:
        ClusteringResult object
    """
    service = ClusteringService(
        algorithm=algorithm,
        distance_threshold=distance_threshold,
    )

    return service.cluster_events(
        events=events,
        embeddings=embeddings,
        documents=documents,
        n_clusters=n_clusters,
    )


def deduplicate_events(
    events: List[Event],
    embeddings: List[Embedding],
    distance_threshold: float = 0.2,
) -> Tuple[List[Event], ClusteringResult]:
    """
    Deduplicate events and return canonical events.

    Args:
        events: List of Event objects
        embeddings: List of Embedding objects
        distance_threshold: Distance threshold for deduplication

    Returns:
        Tuple of (canonical_events, clustering_result)
    """
    result = cluster_events(
        events=events,
        embeddings=embeddings,
        algorithm=ClusteringAlgorithm.AGGLOMERATIVE,
        distance_threshold=distance_threshold,
    )

    # Extract canonical events
    canonical_event_ids = [cluster.centroid_event_id for cluster in result.clusters]
    canonical_events = [e for e in events if e.id in canonical_event_ids]

    return canonical_events, result
