# Deployment notes

Operational guidance for production deployments of cell-explorer-py and its dependencies. This file is intentionally short — only items that aren't obvious from code or local-dev configuration belong here.

## CloudFront / CDN configuration for zarr buckets

Datasets are fetched **directly from storage** by both the browser (frontend) and the server (chat agent via `zarr-access`). When zarr data is hosted in an S3 bucket fronted by CloudFront, the distribution **must vary on the `Origin` request header** or the cache will be poisoned by server-side fetches and break the browser.

### Why

- S3 only emits `Access-Control-Allow-Origin` headers when the request **includes an `Origin` header**. Server-side HTTP clients (aiohttp, httpx, curl) don't send `Origin` automatically — only browsers do.
- CloudFront's default cache key does **not** include the `Origin` header. Same path → same cache entry, regardless of whether the request had `Origin`.
- Sequence that breaks browsers:
  1. Server-side fetch (no Origin) → S3 returns response with **no CORS headers**
  2. CloudFront caches that response
  3. Browser fetch (with Origin) hits the cache → gets the no-CORS response
  4. Browser blocks: *"No 'Access-Control-Allow-Origin' header is present"*

### Required CloudFront config

Use **one of**:

- AWS-managed `CORS-S3Origin` request policy + a response-headers policy with `Origin` configured in CORS, **or**
- A custom cache policy that includes `Origin` in the cache key.

Either makes CloudFront treat browser-Origin and server-no-Origin requests as separate cache entries, so they can't cross-contaminate.

### Current workaround

We don't control the `cbioportal-public-imaging.assets.cbioportal.org` distribution, so `zarr-access` always sends an `Origin` header on server-side requests (default `https://cell-explorer.cbioportal.org`, override per deployment via `ZARR_ACCESS_ORIGIN`). This guarantees S3 emits CORS headers and CloudFront caches a browser-friendly response.

**When standing up a CloudFront distribution we control, fix the cache policy first** rather than relying on the `Origin` injection workaround. The workaround is fragile — any other tool in the ecosystem hitting the same CDN without `Origin` will re-poison the cache.

### Bucket CORS policy

The buckets we use should have `AllowedOrigins: ['*']` for public datasets so any browser origin works. S3 echoes `Access-Control-Allow-Origin: *` regardless of the request Origin value when this is set. If a bucket restricts to specific origins, callers must set `ZARR_ACCESS_ORIGIN` to a value in that allowlist.

## Session lifetimes (Keycloak)

The realm's session timeouts gate when users are forced to re-authenticate. The relevant settings on the `cell-explorer` realm are:

- `ssoSessionIdleTimeout` — current 8h. Idle longer than this and refresh fails (forced re-login).
- `ssoSessionMaxLifespan` — current 24h. Hard cap regardless of activity.
- `accessTokenLifespan` — 5min. Backend refreshes silently before expiry.

Cookie lifetimes are configurable via `ACCESS_COOKIE_MAX_AGE` and `REFRESH_COOKIE_MAX_AGE` (see `.env.example`). Refresh cookie should be `<=` the realm's `ssoSessionMaxLifespan`, otherwise refresh fails before the cookie expires.

## Admin API

`ADMIN_API_KEY` enables admin endpoints (`/api/admin/datasets`, etc.). Required for managing the dataset catalog (create/list/update/delete datasets and datasources) without going through Keycloak admin role.

## Post-migration backfill: obs facet values

After deploying the dataset-facet-values migration, every existing `DatasetMetadata` row has `obs_facets` NULL, so `/api/datasets` reports no facets for the whole catalogue until a refresh runs.

Run this once after deploying, and re-run it until `refreshed` comes back at or near zero:

```bash
curl -s -X POST https://cell-explorer.cbioportal.org/api/admin/datasets/metadata/refresh \
  -H "Authorization: Bearer $ADMIN_API_KEY" -H 'Content-Type: application/json' \
  -d '{"only_stale": true}'
```

**Use `only_stale: true`, not `false`.** For a backfill the two select the same datasets — nothing has been harvested, so everything is stale — but `true` is resumable and `false` is not. The sweep commits per dataset and a successful harvest sets `fetched_at`, so a re-run skips what already succeeded and continues from where it stopped. That matters because the sweep is sequential and takes roughly two seconds per dataset, which can exceed an ingress idle timeout: the request returns 504 while the work continues and persists server-side. Re-running converges. `only_stale: false` re-harvests everything on every attempt and never converges.

Reach for `only_stale: false` only when you deliberately want to re-read datasets whose metadata is already fresh — after fixing an extractor bug, for example.

Two alternatives that avoid the gateway timeout entirely, if you want the whole per-dataset report in one response:

- `kubectl exec` into the pod and curl `http://localhost:8000`, which also keeps `ADMIN_API_KEY` inside the cluster since it resolves from the pod's own environment.
- Call `POST /api/admin/datasets/{slug}/metadata/refresh` per dataset in a loop. Each call is short, and you see progress as it goes.

Expect a small number of `status: "error"` results. Datasets whose store cannot be read are recorded as errors and do not fail the sweep.
