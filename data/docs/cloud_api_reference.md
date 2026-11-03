# Lumora Cloud API Reference (v2)

*Fictional sample document used for the Document Q&A with RAG demo.*

## 1. Base URL and Versioning

All requests are made to `https://api.lumora.example/v2`. The API version is part of the path. Version v1 was deprecated on January 15, 2026 and will be shut down on July 1, 2027. New integrations must use v2.

## 2. Authentication

The API uses bearer tokens. Create an API key in the Developer Console under Settings > API Keys, then send it in the `Authorization` header:

```
Authorization: Bearer <your-api-key>
```

API keys are scoped to a single organization. Each key can be given read-only or read-write scope. Keys that have not been used for 180 days are automatically revoked. For server-to-server integrations that need to act on behalf of users, use the OAuth 2.0 client-credentials flow; access tokens issued by the OAuth flow expire after 60 minutes.

## 3. Rate Limits

Rate limits are applied per API key:

- Free plan: 60 requests per minute
- Pro plan: 600 requests per minute
- Enterprise plan: 3,000 requests per minute, with custom limits available on request

Every response includes the headers `X-RateLimit-Limit`, `X-RateLimit-Remaining`, and `X-RateLimit-Reset`. When a client exceeds its limit, the API returns HTTP 429 with a `Retry-After` header that indicates how many seconds to wait. Clients should retry with exponential backoff.

## 4. Pagination

List endpoints use cursor-based pagination. Responses contain a `next_cursor` field; pass it as the `cursor` query parameter to fetch the next page. The default page size is 50 and the maximum page size is 200, set through the `limit` parameter. When `next_cursor` is null, there are no more results.

## 5. Endpoints

### Devices

- `GET /devices` — list all devices in the organization.
- `GET /devices/{device_id}` — get one device, including its current temperature reading and firmware version.
- `PATCH /devices/{device_id}` — update a device's name, room, or target temperature.
- `POST /devices/{device_id}/commands` — send a command such as `reboot` or `set_mode`.

### Telemetry

- `GET /devices/{device_id}/telemetry` — returns historical readings. The `resolution` parameter accepts `1m`, `15m`, `1h`, or `1d`. Telemetry data is retained for 13 months.

### Webhooks

- `POST /webhooks` — register a webhook URL for events such as `device.offline`, `device.alert`, and `firmware.updated`.

Webhook payloads are signed using HMAC-SHA256. The signature is sent in the `X-Lumora-Signature` header and is computed over the raw request body using your webhook secret. Lumora retries failed webhook deliveries up to 8 times over 24 hours with exponential backoff. Your endpoint must respond with a 2xx status within 10 seconds or the delivery is considered failed.

## 6. Errors

Errors are returned as JSON with a `code` and a human-readable `message`.

| HTTP status | Code | Meaning |
|-------------|------|---------|
| 400 | `invalid_request` | The request body or parameters are malformed. |
| 401 | `unauthenticated` | The API key is missing, invalid, or revoked. |
| 403 | `forbidden` | The key does not have the required scope. |
| 404 | `not_found` | The resource does not exist or belongs to another organization. |
| 409 | `conflict` | The device is busy processing another command. |
| 429 | `rate_limited` | Too many requests. See the Retry-After header. |
| 500 | `internal_error` | Unexpected server error. Retry with backoff. |

## 7. SDKs

Official SDKs are available for Python (`pip install lumora-sdk`) and JavaScript (`npm install @lumora/sdk`). Both SDKs handle pagination and retries on 429 responses automatically.
