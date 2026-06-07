from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

import structlog

from app.core.exceptions import UpstreamServiceError

if TYPE_CHECKING:
    from azure.storage.blob import BlobServiceClient

logger = structlog.get_logger(__name__)


class BlobStorageService:
    """Wraps Azure Blob Storage for resume PDF upload and SAS URL generation.

    When AZURE_STORAGE_CONNECTION_STRING is empty the service is disabled and
    all operations are no-ops — the API layer falls back to local disk instead.
    This lets the app run locally without any Azure credentials.
    """

    def __init__(self, connection_string: str, container: str) -> None:
        self._container = container
        self._available = bool(connection_string)
        self._client: BlobServiceClient | None = None
        self._account_name: str | None = None
        self._account_key: str | None = None

        if self._available:
            try:
                from azure.storage.blob import BlobServiceClient as _BlobServiceClient
                self._client = _BlobServiceClient.from_connection_string(connection_string)
                self._account_name = self._client.account_name
                # account_key is only available when using connection-string auth
                self._account_key = self._client.credential.account_key  # type: ignore[union-attr]
                logger.info(
                    "blob_storage_ready",
                    account=self._account_name,
                    container=container,
                )
            except ImportError:
                logger.warning(
                    "blob_storage_unavailable",
                    reason="azure-storage-blob not installed — pip install azure-storage-blob",
                )
                self._available = False
            except Exception as exc:
                logger.warning("blob_storage_init_failed", error=str(exc))
                self._available = False

    @property
    def available(self) -> bool:
        return self._available

    async def upload(self, blob_name: str, data: bytes) -> None:
        """Upload raw bytes as a blob. No-op when storage is not configured.

        Raises:
            UpstreamServiceError: if the upload call fails.
        """
        if not self._available or self._client is None:
            logger.debug("blob_upload_skipped", blob_name=blob_name, reason="not_configured")
            return

        try:
            from azure.storage.blob import ContentSettings
            blob_client = self._client.get_blob_client(
                container=self._container, blob=blob_name
            )
            # Set content_type and inline disposition so browsers render PDFs instead of downloading
            content_settings = ContentSettings(
                content_type="application/pdf",
                content_disposition="inline",
            )
            await asyncio.to_thread(
                blob_client.upload_blob,
                data,
                overwrite=True,
                content_settings=content_settings,
            )
            logger.info("blob_uploaded", blob_name=blob_name, bytes=len(data))
        except Exception as exc:
            logger.error("blob_upload_failed", blob_name=blob_name, error=str(exc))
            raise UpstreamServiceError(
                "azure_blob", f"Upload failed for '{blob_name}': {exc}"
            ) from exc

    async def delete(self, blob_name: str) -> bool:
        """Delete a blob. Returns True if deleted, False if not found or storage not configured.

        Raises:
            UpstreamServiceError: if the delete call fails for reasons other than not found.
        """
        if not self._available or self._client is None:
            logger.debug("blob_delete_skipped", blob_name=blob_name, reason="not_configured")
            return False

        try:
            blob_client = self._client.get_blob_client(
                container=self._container, blob=blob_name
            )
            await asyncio.to_thread(blob_client.delete_blob)
            logger.info("blob_deleted", blob_name=blob_name)
            return True
        except Exception as exc:
            error_str = str(exc).lower()
            if "blobnotfound" in error_str or "not found" in error_str or "404" in error_str:
                logger.info("blob_delete_not_found", blob_name=blob_name)
                return False
            logger.error("blob_delete_failed", blob_name=blob_name, error=str(exc))
            raise UpstreamServiceError(
                "azure_blob", f"Delete failed for '{blob_name}': {exc}"
            ) from exc

    async def get_sas_url(self, blob_name: str, expiry_minutes: int = 60) -> str | None:
        """Generate a time-limited SAS URL for reading a blob.

        Returns None when storage is not configured (caller falls back to local disk).
        Raises:
            UpstreamServiceError: if SAS generation fails unexpectedly.
        """
        if not self._available or self._client is None:
            return None

        try:
            from azure.storage.blob import BlobSasPermissions, generate_blob_sas

            expiry = datetime.now(timezone.utc) + timedelta(minutes=expiry_minutes)
            sas_token = await asyncio.to_thread(
                generate_blob_sas,
                account_name=self._account_name,
                container_name=self._container,
                blob_name=blob_name,
                account_key=self._account_key,
                permission=BlobSasPermissions(read=True),
                expiry=expiry,
                # Force inline display in browser — prevents download prompt
                content_type="application/pdf",
                content_disposition="inline",
            )
            url = (
                f"https://{self._account_name}.blob.core.windows.net"
                f"/{self._container}/{blob_name}?{sas_token}"
            )
            logger.debug(
                "blob_sas_generated",
                blob_name=blob_name,
                expiry_minutes=expiry_minutes,
            )
            return url
        except Exception as exc:
            logger.error("blob_sas_failed", blob_name=blob_name, error=str(exc))
            raise UpstreamServiceError(
                "azure_blob", f"SAS generation failed for '{blob_name}': {exc}"
            ) from exc
