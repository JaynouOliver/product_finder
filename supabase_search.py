"""
Supabase similarity search module for Product Finder V2.
Uses direct PostgreSQL connection with pgvector for fastest latency.
"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

from config import get_secret


def get_db_config() -> Dict[str, str]:
    """Get database config from environment or Streamlit secrets."""
    return {
        'host': get_secret('DB_HOST'),
        'port': get_secret('DB_PORT', '5432'),
        'dbname': get_secret('DB_NAME', 'postgres'),
        'user': get_secret('DB_USER', 'postgres'),
        'password': get_secret('DB_PASSWORD'),
    }


# Database Configuration (lazy loaded)
DB_CONFIG = None


@dataclass
class SearchResult:
    """Represents a product search result."""
    product_id: str
    name: str
    supplier: Optional[str]
    product_group_id: Optional[str]
    product_type: str
    thumbnail_url: Optional[str]
    similarity: float
    application: Optional[List[str]]
    region_served: Optional[List[str]]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SearchResult':
        return cls(
            product_id=data.get('product_id', ''),
            name=data.get('name', ''),
            supplier=data.get('supplier'),
            product_group_id=data.get('product_group_id'),
            product_type=data.get('product_type', ''),
            thumbnail_url=data.get('thumbnail_url'),
            similarity=data.get('similarity', 0.0),
            application=data.get('application'),
            region_served=data.get('region_served')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'product_id': self.product_id,
            'name': self.name,
            'supplier': self.supplier,
            'product_group_id': self.product_group_id,
            'product_type': self.product_type,
            'thumbnail_url': self.thumbnail_url,
            'similarity': self.similarity,
            'application': self.application,
            'region_served': self.region_served
        }


@dataclass
class SlotResult:
    """Represents search results for a single slot (e.g., Floor, Wall)."""
    slot_name: str
    confidence: float
    results: List[SearchResult]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'slot_name': self.slot_name,
            'confidence': self.confidence,
            'results': [r.to_dict() for r in self.results]
        }


class SupabaseSearch:
    """Direct PostgreSQL similarity search using pgvector HNSW index."""
    
    def __init__(self, db_config: Dict[str, str] = None):
        self.db_config = db_config or get_db_config()
        
        if not self.db_config.get('password'):
            raise ValueError("DB_PASSWORD is required. Set in .env or Streamlit secrets.")
        
        # Keep a persistent connection
        self._conn = None
    
    def _get_conn(self):
        """Get or create a persistent connection."""
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(**self.db_config)
        return self._conn
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connection (reuses connection)."""
        conn = self._get_conn()
        try:
            yield conn
        except Exception:
            # Reset connection on error
            self._conn = None
            raise
    
    def search_by_embedding(
        self,
        embedding: List[float],
        application: Optional[str] = None,
        region: Optional[List[str]] = None,
        price: Optional[int] = None,
        match_count: int = 3,
        similarity_threshold: float = 0.5
    ) -> List[SearchResult]:
        """
        Search for similar products using a raw embedding vector.
        Uses HNSW index for fast approximate nearest neighbor search.
        
        Query Strategy (V2 optimized):
        - Uses CTE to leverage HNSW index FIRST for speed (vector retrieval)
        - Over-fetches 10x results to account for metadata filtering
        - Then applies metadata filters (region, price, application)
        
        Note: V2 spec says "metadata first" but vector-first with over-fetch
        is faster for pgvector HNSW indexes while achieving similar results.
        """
        embedding_str = '[' + ','.join(str(x) for x in embedding) + ']'
        
        # Over-fetch factor to ensure enough results after filtering
        over_fetch = match_count * 10
        
        # Optimized query: Use CTE to leverage HNSW index first, then filter
        query = """
            WITH vector_matches AS (
                SELECT 
                    pe.product_id,
                    pe.embedding <=> %s::vector AS distance
                FROM product_embeddings pe
                WHERE pe.embedding IS NOT NULL
                ORDER BY pe.embedding <=> %s::vector
                LIMIT %s
            )
            SELECT 
                p.id AS product_id,
                p.name,
                p.supplier,
                p.product_group_id,
                p."productType" AS product_type,
                COALESCE(
                    p."materialData"->'files'->>'color_original',
                    p."materialData"->>'renderedImage',
                    p.mesh->>'rendered_image'
                ) AS thumbnail_url,
                (1 - vm.distance)::double precision AS similarity,
                p.application,
                p.metadata->'regionServed' AS region_served
            FROM vector_matches vm
            INNER JOIN "productsV2" p ON vm.product_id = p.id
            WHERE 
                p."objectStatus" IN ('APPROVED', 'APPROVED_PRO')
                AND (1 - vm.distance) >= %s
        """
        
        params = [embedding_str, embedding_str, over_fetch, similarity_threshold]
        
        if application:
            query += """
                AND (
                    p.application IS NULL
                    OR p.application @> %s::jsonb
                    OR p.application @> '["N/A"]'::jsonb
                )
            """
            params.append(f'["{application}"]')
        
        if region:
            query += """
                AND (
                    p.metadata->'regionServed' IS NULL
                    OR p.metadata->'regionServed' ?| %s
                )
            """
            params.append(region)
        
        if price is not None:
            query += " AND (p.metadata->>'relativePrice')::int = %s "
            params.append(price)
        
        query += " ORDER BY vm.distance LIMIT %s "
        params.append(match_count)
        
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
        
        return [SearchResult.from_dict(dict(row)) for row in rows]
    
    def search_slot(
        self,
        slot_name: str,
        embedding: List[float],
        confidence: float,
        region: Optional[List[str]] = None,
        price: Optional[int] = None,
        match_count: int = 3,
        min_confidence: float = 0.3
    ) -> Optional[SlotResult]:
        """Search for products matching a specific slot."""
        if confidence < min_confidence:
            return None
        
        # Map slot names to application categories in the database
        # None = no filter (search all categories)
        application_map = {
            # Surfaces
            'floor': 'Floors', 'floors': 'Floors',
            'wall': 'Walls', 'walls': 'Walls',
            'worktop': 'Worktops', 'worktop / surface': 'Worktops',
            'backsplash': 'Backsplash',
            'countertop': 'Countertops', 'ceiling': 'Ceilings', 'ceilings': 'Ceilings',
            # Fabrics
            'upholstery': 'Upholstery',
            'curtain': 'Curtain', 'curtains': 'Curtain',
            'sofa': 'Upholstery', 'carpet': 'Floors', 'rug': 'Floors',
            'outdoor fabric': 'Outdoor Fabric',
            # Decor (no filter - search all)
            'furniture': None,
            'decor': 'Decor',
            'paintings': None,  # No specific category, search all
            'wallpaper': 'Wallpaper / Wallcovering',
            'wallpaper / wallcovering': 'Wallpaper / Wallcovering',
            # Hardware
            'fixtures': 'Fixtures',
            'faucet / tap': 'Faucet / Tap',
            'handle': 'Handle',
            'knob': 'Knob',
            'switch': 'Switch',
        }
        
        application = application_map.get(slot_name.lower())
        
        # Lower similarity threshold for better recall
        # Base: 0.3, max boost: 0.15 (so max threshold = 0.45)
        similarity_threshold = 0.3 + (confidence * 0.15)
        
        results = self.search_by_embedding(
            embedding=embedding,
            application=application,
            region=region,
            price=price,
            match_count=match_count,
            similarity_threshold=similarity_threshold
        )
        
        return SlotResult(slot_name=slot_name, confidence=confidence, results=results)
    
    def search_all_slots_parallel(
        self,
        slots: List[Dict[str, Any]],
        region: Optional[List[str]] = None,
        price: Optional[int] = None,
        match_count: int = 3,
        min_confidence: float = 0.3,
        max_workers: int = 4
    ) -> Dict[str, SlotResult]:
        """Search all slots in parallel for maximum speed."""
        results = {}
        
        def search_single_slot(slot: Dict[str, Any]) -> tuple:
            result = self.search_slot(
                slot_name=slot['name'],
                embedding=slot['embedding'],
                confidence=slot['confidence'],
                region=region,
                price=price,
                match_count=match_count,
                min_confidence=min_confidence
            )
            return (slot['name'], result)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(search_single_slot, slot) for slot in slots]
            for future in futures:
                name, result = future.result()
                if result is not None:
                    results[name] = result
        
        return results
    
    def search_fallback(self, embedding: List[float], match_count: int = 12) -> List[SearchResult]:
        """Fallback search: No metadata filters, just pure similarity."""
        return self.search_by_embedding(
            embedding=embedding,
            match_count=match_count,
            similarity_threshold=0.3
        )
    
    def test_connection(self) -> bool:
        """Test database connection."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT NOW()")
                    print(f"Connection successful! Server time: {cur.fetchone()[0]}")
                    return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False


if __name__ == "__main__":
    search = SupabaseSearch()
    search.test_connection()
