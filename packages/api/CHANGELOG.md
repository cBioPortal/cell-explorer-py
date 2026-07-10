# CHANGELOG

<!-- version list -->

## v0.3.0 (2026-07-10)

### Chores

- **api**: Declare cryptography dependency explicitly
  ([`f5f3030`](https://github.com/cBioPortal/cell-explorer-py/commit/f5f303066a0c1ccac129e8e47c570c08aecdea82))

### Features

- **api**: Implement CloudFront signed-cookie minting
  ([`2538e0c`](https://github.com/cBioPortal/cell-explorer-py/commit/2538e0c0f005237ca0f98cced4fe33d652f69e38))

- **api**: Pass bedrock_region when constructing the LLM client
  ([`d533e1b`](https://github.com/cBioPortal/cell-explorer-py/commit/d533e1b16e6c1245cdef64169b4a94f42e20f35d))

- **tools**: Describe_var_column + var_columns in dataset schema
  ([`c578231`](https://github.com/cBioPortal/cell-explorer-py/commit/c578231426d0307c841bc8c4c39ae60a60337ce7))

- **zarr-adapter**: Recognize 'gene' as a gene-symbol column candidate
  ([`42ef311`](https://github.com/cBioPortal/cell-explorer-py/commit/42ef311456317a4685718ccb1b798f2f68cce3b3))

### Testing

- **api**: Harden cloudfront tests (expiry range, partial-env) + catch UnsupportedAlgorithm
  ([`f57d210`](https://github.com/cBioPortal/cell-explorer-py/commit/f57d210deb58b481561b917889382c4c9594e338))


## v0.2.0 (2026-05-21)

### Bug Fixes

- Add CORS headers to CLI callback server for cross-origin POST
  ([`9aa8df9`](https://github.com/cBioPortal/cell-explorer-py/commit/9aa8df9c28907526bbf6d2a893ccbec244d79fc9))

- Add response_model to auth endpoints for OpenAPI spec
  ([`ca32c17`](https://github.com/cBioPortal/cell-explorer-py/commit/ca32c1715810e1eb0baa98d21a8a95ae26479131))

- Always register auth routes (501 when disabled) and guard SPA catch-all for /api/* paths
  ([`0aa198c`](https://github.com/cBioPortal/cell-explorer-py/commit/0aa198ca4899745f3ab9f675c61b5db89b770bdd))

- Respect X-Forwarded-Host/Proto headers in auth redirect URIs
  ([`b1b0baa`](https://github.com/cBioPortal/cell-explorer-py/commit/b1b0baa5cfeb9b24a1d0c775b3a726abe66ca61b))

- Revert to standard aud validation with Keycloak audience mapper
  ([`261100a`](https://github.com/cBioPortal/cell-explorer-py/commit/261100ad619809316c3da3faefe508cbf80769b4))

- Validate azp instead of aud, disable secure cookies on localhost
  ([`a5128f1`](https://github.com/cBioPortal/cell-explorer-py/commit/a5128f186b5f9f9bbef8b869672c371f358c4e9b))

- **api**: Add PUT, DELETE and Authorization header to CORS config
  ([`87c5e7a`](https://github.com/cBioPortal/cell-explorer-py/commit/87c5e7a3d6a79506eb6cd556a5173b597c493a1c))

- **api**: Address code review findings
  ([`74e3b48`](https://github.com/cBioPortal/cell-explorer-py/commit/74e3b480d84ca2b81aa65eae30c33513f6e7e736))

- **api**: Align prompt_addendum order with ORM + tighten test coverage
  ([`507531a`](https://github.com/cBioPortal/cell-explorer-py/commit/507531a286ebf12a71d9795d54eaffead96f62b5))

- **api**: Prompt_addendum column uses sa.Text per code review
  ([`65af09f`](https://github.com/cBioPortal/cell-explorer-py/commit/65af09faabcf32d84d1e85d40540da7f7ede4047))

- **api**: Read database URL from app settings in alembic env.py
  ([`e94d04d`](https://github.com/cBioPortal/cell-explorer-py/commit/e94d04da95e836706448c955d6010b1cca9e7991))

- **api**: Set cookie secure flag based on request protocol, not hostname
  ([`9063ef7`](https://github.com/cBioPortal/cell-explorer-py/commit/9063ef77bc8152fd8e9e25e45147980f75dfae00))

- **api**: Set refreshed token cookies on /me response
  ([`50820f4`](https://github.com/cBioPortal/cell-explorer-py/commit/50820f4d07726cd9fb4936d65b211d29f1f9b777))

- **api**: Standardize chat feedback timestamps via _utcnow
  ([`d295eec`](https://github.com/cBioPortal/cell-explorer-py/commit/d295eec95348675bc6ccbc9923809a4c854a83d1))

- **auth**: JWT leeway + pure-ASGI token refresh middleware
  ([`18c0a1f`](https://github.com/cBioPortal/cell-explorer-py/commit/18c0a1fd8789267b1d15dd22ba981c903d6bb8d8))

- **auth**: Refresh tokens when access cookie is missing
  ([#68](https://github.com/cBioPortal/cell-explorer-py/pull/68),
  [`0d15c97`](https://github.com/cBioPortal/cell-explorer-py/commit/0d15c978d2e88a0f0feceb086e1e53e6956854a2))

- **chat**: Address review feedback for A4
  ([`60e962a`](https://github.com/cBioPortal/cell-explorer-py/commit/60e962a7082ed984419a868a4ea7f25d2ef2ae60))

- **chat**: Persist assistant placeholder when turn fails
  ([`de0960f`](https://github.com/cBioPortal/cell-explorer-py/commit/de0960f8b4f421d5764ba5235e1fd96f8b15c722))

- **cli**: Resolve decode_column return types and user display name
  ([`f0b61ef`](https://github.com/cBioPortal/cell-explorer-py/commit/f0b61ef5a88e6853a62e95efc1ebe9ec7755be01))

- **cli**: Use a single asyncio.Runner for the whole REPL session
  ([`f53072b`](https://github.com/cBioPortal/cell-explorer-py/commit/f53072bbb313f5fe258f3c75d0eff62f8656f9e7))

- **cli**: Use asyncio.Runner for 'ask' command (matches REPL fix)
  ([#59](https://github.com/cBioPortal/cell-explorer-py/pull/59),
  [`c8c3455`](https://github.com/cBioPortal/cell-explorer-py/commit/c8c3455756b47a27060f9a62e736cde4330e8a2a))

- **cli) + feat(zarr-access**: Asyncio.Runner for 'ask', plus opt-in HTTP tracing
  ([#59](https://github.com/cBioPortal/cell-explorer-py/pull/59),
  [`c8c3455`](https://github.com/cBioPortal/cell-explorer-py/commit/c8c3455756b47a27060f9a62e736cde4330e8a2a))

- **db**: Enable SQLite foreign_keys pragma on every connection
  ([`8acb356`](https://github.com/cBioPortal/cell-explorer-py/commit/8acb356b0307d4000b27969eae14d81c15f5b9c6))

- **threads**: Address review feedback for A3
  ([`6979c55`](https://github.com/cBioPortal/cell-explorer-py/commit/6979c55504d89e7fcde0a73c7e72c8f748454246))

- **zarr_adapter**: Auto-detect gene symbol column, do column-wise X reads
  ([`1a9b51a`](https://github.com/cBioPortal/cell-explorer-py/commit/1a9b51a89e3a2f64be9563595fc5207ed581ccd2))

### Chores

- Add Makefile with common dev commands
  ([`15b8da9`](https://github.com/cBioPortal/cell-explorer-py/commit/15b8da96c403f15020f8ed1b472d40a301c0b6ea))

- Add OpenAPI tags to group endpoints by section
  ([`3a113da`](https://github.com/cBioPortal/cell-explorer-py/commit/3a113da8d73b021b6905023adc98d304fecce09e))

- Add pyjwt and httpx dependencies for auth
  ([`129e787`](https://github.com/cBioPortal/cell-explorer-py/commit/129e787426f3df0e38b93a15e2c9d4ce45f88e8f))

- Keep semantic-release in 0.x range
  ([`3b49bbd`](https://github.com/cBioPortal/cell-explorer-py/commit/3b49bbdd2ccbee1c1dbe8ce3072d1a1af2a1a069))

- Switch pytest to importlib import mode for monorepo
  ([`b81b6b2`](https://github.com/cBioPortal/cell-explorer-py/commit/b81b6b29d0a26b7d93df2d26a212e8b0012439c1))

- **api**: Declare cell-explorer-agent and zarr-access as workspace deps
  ([#66](https://github.com/cBioPortal/cell-explorer-py/pull/66),
  [`e2fe9d0`](https://github.com/cBioPortal/cell-explorer-py/commit/e2fe9d0e48606023130d5a2984ae6f6bc948bb04))

### Documentation

- **api**: Clarify cache-key invariant covers prompt_addendum
  ([`11eb1e0`](https://github.com/cBioPortal/cell-explorer-py/commit/11eb1e07748c464e13c9c68a71bb6cd291283e43))

### Features

- Add /api/auth/cli-login endpoint with signed state
  ([`0318a2e`](https://github.com/cBioPortal/cell-explorer-py/commit/0318a2e6d9330d2e02c1668652e2a57f5c96c667))

- Add app data directory and rotating file logging
  ([`ba6df83`](https://github.com/cBioPortal/cell-explorer-py/commit/ba6df8320fab2ef9e36c933fae10de42368cc312))

- Add auth routes (login, callback, me, logout, token-exchange)
  ([`74279dc`](https://github.com/cBioPortal/cell-explorer-py/commit/74279dca66f3f0a32f8f1c0b0ade3ab4eb46bace))

- Add Keycloak and CORS settings to config
  ([`feb65ed`](https://github.com/cBioPortal/cell-explorer-py/commit/feb65eddea939dd251119a25ab37c6dc6d7cd4ff))

- Add Keycloak OIDC client with token validation
  ([`5803a59`](https://github.com/cBioPortal/cell-explorer-py/commit/5803a59aa41606f992bf95f748984e16002e4ce2))

- Add require_auth dependency for protected routes
  ([`4015448`](https://github.com/cBioPortal/cell-explorer-py/commit/40154483ad65f61754198cc7c3cd82fd417324e4))

- Add StrataZarrAccess adapter for strata-protocol
  ([`29b29a6`](https://github.com/cBioPortal/cell-explorer-py/commit/29b29a678a978c38f2f28aa108adca6e295ca52f))

- Add User model for auth
  ([`17616ed`](https://github.com/cBioPortal/cell-explorer-py/commit/17616ed20f7fe67acba386aaa2bc57a112bc0c35))

- AnnDataZarrAccess gene methods and obs_mask
  ([`3a71f43`](https://github.com/cBioPortal/cell-explorer-py/commit/3a71f433837ab30e6435b681a8c91731bc788d30))

- AnnDataZarrAccess obs_column + obs_columns translation
  ([`fcc8212`](https://github.com/cBioPortal/cell-explorer-py/commit/fcc82128858fa2fc035f98bfc9f000c4fce9dbf7))

- AnnDataZarrAccess skeleton (shape/attrs/obsm_keys/var_names)
  ([`d7bec10`](https://github.com/cBioPortal/cell-explorer-py/commit/d7bec10affdecb61df91195da48df4216af8ba7b))

- Branch /api/auth/callback on signed CLI state; return token HTML
  ([`883b646`](https://github.com/cBioPortal/cell-explorer-py/commit/883b646461006880ad2e32ab7360a03638184f63))

- CLI access-token refresh via Keycloak
  ([`26e8220`](https://github.com/cBioPortal/cell-explorer-py/commit/26e8220977f90ec62a816fea345ffa3dc3d1a368))

- CLI ask command (one-shot chat)
  ([`b0f8c93`](https://github.com/cBioPortal/cell-explorer-py/commit/b0f8c93f5e6bb9c0780cb9af0d2dc9340d7f02ff))

- CLI auth.json load/save at mode 0600
  ([`337855f`](https://github.com/cBioPortal/cell-explorer-py/commit/337855f43efdfa8cc86638ebef6fabede2b65919))

- CLI datasets command
  ([`456fc16`](https://github.com/cBioPortal/cell-explorer-py/commit/456fc162f1cd8264e443437df777d7af66b4425d))

- CLI event renderer
  ([`9c5787f`](https://github.com/cBioPortal/cell-explorer-py/commit/9c5787f25cb3d0bbd8a0d298f0d390d2140fcb5e))

- CLI local callback server for OAuth flow
  ([`c3b8266`](https://github.com/cBioPortal/cell-explorer-py/commit/c3b8266f55abce9984e5e93073992e362b4e3d89))

- CLI logout command
  ([`60f89bd`](https://github.com/cBioPortal/cell-explorer-py/commit/60f89bd26597ec82355e6a6a096f0a188d83e024))

- CLI repl command
  ([`e21b8db`](https://github.com/cBioPortal/cell-explorer-py/commit/e21b8dbf80b53f5a0dfd6756acbe9a1618c0ada8))

- CLI scaffold with login command
  ([`fc236cf`](https://github.com/cBioPortal/cell-explorer-py/commit/fc236cf2a1d7c261b71add3343d2d389333a6c12))

- Expose auth_enabled in /api/info response
  ([`88d4cf7`](https://github.com/cBioPortal/cell-explorer-py/commit/88d4cf7019c0095c2357049836b251c7ba09997c))

- Extract user_can_access into services/access.py
  ([`7e70d21`](https://github.com/cBioPortal/cell-explorer-py/commit/7e70d21f24eb920aedf0bee750de295706046870))

- Fetch Keycloak JWKS on app startup
  ([`3d488e7`](https://github.com/cBioPortal/cell-explorer-py/commit/3d488e7e492a9e4f0238472d9479e7c5f91b0458))

- Make_chat_agent factory with public path and error types
  ([`c9dd2d5`](https://github.com/cBioPortal/cell-explorer-py/commit/c9dd2d5642993891b2ecc622638512d60baba658))

- Row-selective atomic reads for compare_groups
  ([`ef0118b`](https://github.com/cBioPortal/cell-explorer-py/commit/ef0118b63c05e08bdfc5168fee0d3d4379b6d2b5))

- Scaffold cli/ package with errors module
  ([`f76cca6`](https://github.com/cBioPortal/cell-explorer-py/commit/f76cca68d32b5f913d3fe037d984581e97787938))

- Wire cell-explorer-chat script entry in cell-explorer-api
  ([`23d45b2`](https://github.com/cBioPortal/cell-explorer-py/commit/23d45b27682500e57f0a600d25afb14bc8cc85a9))

- Wire StrataZarrAccess into chat_session catalog construction
  ([`1e86bed`](https://github.com/cBioPortal/cell-explorer-py/commit/1e86bed20b52f11157e152c2c139c42c020210c2))

- **access**: Compute_chat_permission helper for global chat role gate
  ([`403dea1`](https://github.com/cBioPortal/cell-explorer-py/commit/403dea108dfb3da6d465b1ba2d2c1f748d1ee717))

- **admin**: Expose chat_enabled in dataset create/update/response
  ([`ddb71c9`](https://github.com/cBioPortal/cell-explorer-py/commit/ddb71c9e80dc6ebc9350356962355625dc745fa0))

- **agent**: Accept view_state on chat turns
  ([`2aec992`](https://github.com/cBioPortal/cell-explorer-py/commit/2aec992161cda43834bfb825213cbde44869cdfe))

- **agent**: Atomic-strata fast path for compare_groups when no coarse covers obs_column
  ([`53679fd`](https://github.com/cBioPortal/cell-explorer-py/commit/53679fd1ab9405232ee9147da117d14bc8f4b948))

- **agent**: Expose obs column values to ObsColumnInfo
  ([#63](https://github.com/cBioPortal/cell-explorer-py/pull/63),
  [`612528b`](https://github.com/cBioPortal/cell-explorer-py/commit/612528bbdd8f70c5c696a3670313f75d36006f32))

- **agent**: Thread prompt_addendum through DatasetContext
  ([`099a8ec`](https://github.com/cBioPortal/cell-explorer-py/commit/099a8ecb740a321d9bbe2112f6a61230360f70e5))

- **api**: /api/chat/{slug} routes — context + turns NDJSON streaming
  ([#64](https://github.com/cBioPortal/cell-explorer-py/pull/64),
  [`60c05dd`](https://github.com/cBioPortal/cell-explorer-py/commit/60c05ddd1e9b69e00e450fd6c47b2e7ca904f909))

- **api**: Add admin CRUD endpoints for datasources and datasets
  ([`3cf44e4`](https://github.com/cBioPortal/cell-explorer-py/commit/3cf44e4d79b577b15e1f8a29fb5df25216327fc2))

- **api**: Add Alembic setup with initial datasources/datasets migration
  ([`e543715`](https://github.com/cBioPortal/cell-explorer-py/commit/e54371534d00dfbb11c7f5c3cd5b7659688b45fc))

- **api**: Add async database engine and session factory
  ([`e917425`](https://github.com/cBioPortal/cell-explorer-py/commit/e917425f6dc6135d9bafea0ad5bbdccfbb5b6357))

- **api**: Add bearer auth to Swagger UI with persistent authorization
  ([`affface`](https://github.com/cBioPortal/cell-explorer-py/commit/affface8a34d6d9b4c5e981a17e3fa06eaed62d5))

- **api**: Add ChatFeedback SQLModel table
  ([`6719434`](https://github.com/cBioPortal/cell-explorer-py/commit/6719434395ed1a21422cac65223d0c53def817e4))

- **api**: Add credential minting service with http_token and cloudfront support
  ([`ec4db87`](https://github.com/cBioPortal/cell-explorer-py/commit/ec4db87309fe2336771fb795921e096049f6c5f6))

- **api**: Add database seed script for development
  ([`ad6db6f`](https://github.com/cBioPortal/cell-explorer-py/commit/ad6db6fa0177ccba5497aba45970e6ba29da1cf5))

- **api**: Add DATABASE_URL and ADMIN_API_KEY settings
  ([`36cdcc9`](https://github.com/cBioPortal/cell-explorer-py/commit/36cdcc987d9eccea580284c8ca1f39015053c265))

- **api**: Add dataset access endpoint with credential minting
  ([`67eda4a`](https://github.com/cBioPortal/cell-explorer-py/commit/67eda4a6996ab53056e6c3ffc2435412b6d51f5a))

- **api**: Add dataset discovery endpoints with role-based filtering
  ([`b95559b`](https://github.com/cBioPortal/cell-explorer-py/commit/b95559bcd11134728afb36ca19883c249d7376e1))

- **api**: Add Datasource and Dataset SQLModel table definitions
  ([`c2eea15`](https://github.com/cBioPortal/cell-explorer-py/commit/c2eea155ddb80367be58c3eba3ecd9a115901fde))

- **api**: Add datasource internal_base_url for split client/server URLs
  ([`c7e2c2d`](https://github.com/cBioPortal/cell-explorer-py/commit/c7e2c2d101b5f90fb9e5f6254f18bfcc31448e3a))

- **api**: Add GET /admin/datasets to list all datasets
  ([`64b34d2`](https://github.com/cBioPortal/cell-explorer-py/commit/64b34d20fbfb024ddaba5aedcc9c3f1c8ffc2868))

- **api**: Add KEYCLOAK_IDP_HINT env var for kc_idp_hint pass-through
  ([`20ebedd`](https://github.com/cBioPortal/cell-explorer-py/commit/20ebeddccb03a8aa24c281af488822fbcbf7c946))

- **api**: Add nullable prompt_addendum column to Dataset
  ([`dec9990`](https://github.com/cBioPortal/cell-explorer-py/commit/dec99908f81fe57567cc71bdec30efe2dce8beda))

- **api**: Add optional_auth dependency for dataset listing
  ([`5def2fd`](https://github.com/cBioPortal/cell-explorer-py/commit/5def2fd542e940e36a5b36629b242d3aa11045ba))

- **api**: Add require_admin dependency with API key and Keycloak support
  ([`1bef938`](https://github.com/cBioPortal/cell-explorer-py/commit/1bef938ca49511c69ed78982ba47206d704212bb))

- **api**: Add sqlmodel, alembic, aiosqlite dependencies
  ([`bb51a8f`](https://github.com/cBioPortal/cell-explorer-py/commit/bb51a8fb7423dd8caa5526f6c64fa57f28b6f28d))

- **api**: Admin endpoints accept + return prompt_addendum
  ([`6458b21`](https://github.com/cBioPortal/cell-explorer-py/commit/6458b2130529f9541fe4182528e34ec77a16a00f))

- **api**: Alembic migration for chat_feedback
  ([`c7aa2b7`](https://github.com/cBioPortal/cell-explorer-py/commit/c7aa2b75b9c21f8225d78062a362118a2b4834ca))

- **api**: DELETE /chat/.../messages/{id}/feedback
  ([`8fadf1d`](https://github.com/cBioPortal/cell-explorer-py/commit/8fadf1de1fc94350baff6a79591e70a85df97154))

- **api**: Embed caller's feedback in thread detail response
  ([`6fb2053`](https://github.com/cBioPortal/cell-explorer-py/commit/6fb2053f4dbad62d6bb19aa065227db53edfd1cc))

- **api**: Emit in-stream error event on mid-turn agent failure
  ([#64](https://github.com/cBioPortal/cell-explorer-py/pull/64),
  [`60c05dd`](https://github.com/cBioPortal/cell-explorer-py/commit/60c05ddd1e9b69e00e450fd6c47b2e7ca904f909))

- **api**: Expose ObsColumnInfo.values in /context response
  ([#64](https://github.com/cBioPortal/cell-explorer-py/pull/64),
  [`60c05dd`](https://github.com/cBioPortal/cell-explorer-py/commit/60c05ddd1e9b69e00e450fd6c47b2e7ca904f909))

- **api**: Flush Langfuse on lifespan shutdown
  ([`45239ef`](https://github.com/cBioPortal/cell-explorer-py/commit/45239ef1fbe2f895e3e7234b03febf6a7e1456c3))

- **api**: Gate chat on ANTHROPIC_API_KEY presence
  ([#69](https://github.com/cBioPortal/cell-explorer-py/pull/69),
  [`0b8e641`](https://github.com/cBioPortal/cell-explorer-py/commit/0b8e641acbda057608f01ea78684996f02f8399e))

- **api**: GET /api/chat/{slug}/context returns dataset context
  ([#64](https://github.com/cBioPortal/cell-explorer-py/pull/64),
  [`60c05dd`](https://github.com/cBioPortal/cell-explorer-py/commit/60c05ddd1e9b69e00e450fd6c47b2e7ca904f909))

- **api**: Make session cookie lifetimes configurable
  ([#67](https://github.com/cBioPortal/cell-explorer-py/pull/67),
  [`6dcc981`](https://github.com/cBioPortal/cell-explorer-py/commit/6dcc98135aeb47cb67db528b5f81fb2d16358e20))

- **api**: Map ChatSessionError subclasses to HTTP status codes
  ([#64](https://github.com/cBioPortal/cell-explorer-py/pull/64),
  [`60c05dd`](https://github.com/cBioPortal/cell-explorer-py/commit/60c05ddd1e9b69e00e450fd6c47b2e7ca904f909))

- **api**: Merge client roles with realm roles in decoded User
  ([`3cea7f9`](https://github.com/cBioPortal/cell-explorer-py/commit/3cea7f9e15b66fa55dadebba89dd31062d442fac))

- **api**: Pass telemetry_context from chat route into agent
  ([`dc52d54`](https://github.com/cBioPortal/cell-explorer-py/commit/dc52d547b9d7018d1258dfdc7e14a4ad8a80d0ed))

- **api**: POST /api/chat/{slug}/turns request validation
  ([#64](https://github.com/cBioPortal/cell-explorer-py/pull/64),
  [`60c05dd`](https://github.com/cBioPortal/cell-explorer-py/commit/60c05ddd1e9b69e00e450fd6c47b2e7ca904f909))

- **api**: POST /api/chat/{slug}/turns streams agent events as NDJSON
  ([#64](https://github.com/cBioPortal/cell-explorer-py/pull/64),
  [`60c05dd`](https://github.com/cBioPortal/cell-explorer-py/commit/60c05ddd1e9b69e00e450fd6c47b2e7ca904f909))

- **api**: PUT /chat/.../messages/{id}/feedback (happy path)
  ([`6bd931d`](https://github.com/cBioPortal/cell-explorer-py/commit/6bd931d934f13c216adbe09bbc4b995cb2652520))

- **api**: Scaffold chat router ([#64](https://github.com/cBioPortal/cell-explorer-py/pull/64),
  [`60c05dd`](https://github.com/cBioPortal/cell-explorer-py/commit/60c05ddd1e9b69e00e450fd6c47b2e7ca904f909))

- **api**: Switch http_token credential minting from HS256 to RS256
  ([`87ec115`](https://github.com/cBioPortal/cell-explorer-py/commit/87ec115080da9a3cfe98aa6c9a6746c88d9d0cf4))

- **api**: Wire database engine into app lifecycle with get_db dependency
  ([`b2b63d2`](https://github.com/cBioPortal/cell-explorer-py/commit/b2b63d2fad3e19a67f1fdbc5e5056a6201cd645e))

- **auth**: POST /api/auth/refresh endpoint
  ([`c163f7c`](https://github.com/cBioPortal/cell-explorer-py/commit/c163f7c61539c68a74a5b5256e01871708cdbecd))

- **chat**: /context optional_auth + permission field
  ([`944e176`](https://github.com/cBioPortal/cell-explorer-py/commit/944e17649299852fa4e0ec2aa7bb3ce32d821437))

- **chat**: /turns 403 on missing chat role; 404 on dataset chat-disabled
  ([`e5afff6`](https://github.com/cBioPortal/cell-explorer-py/commit/e5afff692f406c1bd4f766cba6597d6596141703))

- **chat**: /turns persists messages and emits thread_open
  ([`0916f13`](https://github.com/cBioPortal/cell-explorer-py/commit/0916f1397d4fe112310cf43ab661b3e4e2291799))

- **chat**: ChatDisabledError + make_chat_agent gate
  ([`129f990`](https://github.com/cBioPortal/cell-explorer-py/commit/129f990ee15266399c927dc10e1a4162ffadfe1e))

- **chat**: DELETE /threads/{id} — hard delete with cascade
  ([`b387014`](https://github.com/cBioPortal/cell-explorer-py/commit/b387014596fc42c4b00ff09c3ff9c064f22a245b))

- **chat**: Emit assistant message_id in done event
  ([`ef776d6`](https://github.com/cBioPortal/cell-explorer-py/commit/ef776d65c5e82e203a6394504bf5f37f9b280421))

- **chat**: Forward 👍/👎 feedback to Langfuse Scores
  ([`5d80616`](https://github.com/cBioPortal/cell-explorer-py/commit/5d80616f7847a8d903fc0b95d65ae17a35ba1a18))

- **chat**: GET /threads — list a user's threads on a dataset
  ([`71fc793`](https://github.com/cBioPortal/cell-explorer-py/commit/71fc793ea4802fc6e181c5a4aec97c96af84a24a))

- **chat**: GET /threads/{id} — load one thread's history
  ([`f95efb2`](https://github.com/cBioPortal/cell-explorer-py/commit/f95efb2da1b7c8eb5017e946dff401f503b3cb1d))

- **config**: Add chat_required_role setting; seed chat-enabled
  ([`e47647b`](https://github.com/cBioPortal/cell-explorer-py/commit/e47647b7a0061d6e7bbd98c8147ff2e487cc371b))

- **datasets**: Expose chat_enabled in DatasetResponse
  ([`87d6003`](https://github.com/cBioPortal/cell-explorer-py/commit/87d60036bc08c0baef0e1510c69d2c89816b9452))

- **db**: Add chat_threads and chat_messages tables
  ([`f859898`](https://github.com/cBioPortal/cell-explorer-py/commit/f85989846478731ef24d0a8371b5ac77e4252da3))

- **db**: Add Dataset.chat_enabled column
  ([`bcfdb7b`](https://github.com/cBioPortal/cell-explorer-py/commit/bcfdb7bdec86abbebe65baeca3bb57c5542f996b))

- **telemetry**: Capture Langfuse trace_id on assistant messages
  ([`b985f2e`](https://github.com/cBioPortal/cell-explorer-py/commit/b985f2eb73a15357a7a36edf1e3a54ca04fed7a5))

- **telemetry**: Use email as Langfuse user_id (fallback to sub)
  ([`27bf54f`](https://github.com/cBioPortal/cell-explorer-py/commit/27bf54f4ac913959a77fec72610dae636b0f332f))

- **threads**: Services/threads.py — create/list/load/append/delete
  ([`eef4091`](https://github.com/cBioPortal/cell-explorer-py/commit/eef4091208fda95762abde97a6349f3063231aa1))

- **zarr-access**: Add HTTP request tracing toggled by env var
  ([#59](https://github.com/cBioPortal/cell-explorer-py/pull/59),
  [`c8c3455`](https://github.com/cBioPortal/cell-explorer-py/commit/c8c3455756b47a27060f9a62e736cde4330e8a2a))

- **zarr-access**: Add var_columns() to ZarrAccess protocol
  ([`4291858`](https://github.com/cBioPortal/cell-explorer-py/commit/42918584ec05853bd8bd1a70b20f8f2a94c8388a))

### Performance Improvements

- **chat**: Cache DatasetContext in chat_session by (slug, updated_at)
  ([`7401cdc`](https://github.com/cBioPortal/cell-explorer-py/commit/7401cdcd778fa9f94c5ea702550c01d2291b4814))

- **chat**: Lightweight gating for /threads endpoints
  ([`e7caa38`](https://github.com/cBioPortal/cell-explorer-py/commit/e7caa38eebe659c6922240d6f53bb38091f87616))

- **zarr-access**: Read var_columns from AnnDataStore property
  ([`da83e06`](https://github.com/cBioPortal/cell-explorer-py/commit/da83e06de1c5fdf7e5cc159ff04d26af07a188d0))

### Refactoring

- **api**: Move token refresh cookies to middleware
  ([`0bf5512`](https://github.com/cBioPortal/cell-explorer-py/commit/0bf5512d8ac9c74f08af407f6c7ac0bb97d4e9a8))

- **api**: Use _set_token_cookies in middleware to avoid duplication
  ([`a8b1dc7`](https://github.com/cBioPortal/cell-explorer-py/commit/a8b1dc7123e3f3a8706f96fa39dbf7e7c969148c))

- **test**: Share route-test fixtures via conftest
  ([`13b6348`](https://github.com/cBioPortal/cell-explorer-py/commit/13b6348bd275e5e735d1d9b247a5846610134cdf))

- **test**: Use project create_engine in seeded_app fixture
  ([`4707ee8`](https://github.com/cBioPortal/cell-explorer-py/commit/4707ee87625724acb9add61a2a14633906f05ccc))

### Testing

- Cover make_chat_agent private + credential-failure paths
  ([`c077d46`](https://github.com/cBioPortal/cell-explorer-py/commit/c077d467a454a5457d078e24afa0f9c4d6c8b32d))

- **api**: Add test for refreshed token cookies on /me response
  ([`34828f3`](https://github.com/cBioPortal/cell-explorer-py/commit/34828f30ba0686f59ba8349bfe0fa4c8bc492c32))

- **api**: Assert refreshed cookie values match new tokens
  ([`fdb0aaf`](https://github.com/cBioPortal/cell-explorer-py/commit/fdb0aaf90e3379670718753c269af64bd1fdfd57))

- **api**: Tighten feedback comment isolation assertion
  ([`ac91430`](https://github.com/cBioPortal/cell-explorer-py/commit/ac9143086a42ccf94f91fd24251554bb20fb1abe))

- **api**: Upsert/ownership/validation tests for feedback PUT
  ([`86a6c1d`](https://github.com/cBioPortal/cell-explorer-py/commit/86a6c1d1709bfe1d164cad60fa05e7179866354e))

- **api**: Verify cookie secure flag based on protocol
  ([`07e7737`](https://github.com/cBioPortal/cell-explorer-py/commit/07e7737eee2271051ac77894fa026ca344bebac9))

- **auth**: Use exp=-60 so expired-token tests trip past the 30s leeway
  ([`7cfa513`](https://github.com/cBioPortal/cell-explorer-py/commit/7cfa513a2133a756ed27b1fcdf7aae09c9f4741e))


## v0.1.0 (2026-04-02)

- Initial Release
