# Host BabyInfoQuick on a Linux VM

Use an x86-64 Linux VM with Docker Engine and the Compose plugin. For a small portfolio demo, start with 4 vCPUs, 8 GiB RAM and at least 25 GB of free disk. Allow inbound TCP ports 80 and 443. Point a DNS A record at the VM's public IP. If the name also has an AAAA record, it must reach the same server over IPv6.

## Install

To create a new archive from an indexed checkout, run `python deploy/package.py`. The output is `dist/babyinfoquick-host.tar.gz`. The packager includes the app, benchmark, configuration and public source chunks. It excludes local credentials and query logs.

Upload the deployment archive and run these commands on the VM. The archive includes the exact fixed and section chunks used for the recorded benchmark. A source checkout also works. Without the bundled chunks, setup fetches and parses the source PDFs. Either path requires outbound HTTPS for the Python packages, container images and model weights.

```sh
mkdir -p ~/babyinfoquick
tar -xzf babyinfoquick-host.tar.gz -C ~/babyinfoquick
cd ~/babyinfoquick
cp .env.host.example .env.host
chmod 600 .env.host
nano .env.host
```

Set `SITE_ADDRESS` to your DNS name, for example `demo.yourdomain.com`. Leave the two key fields empty for public passage search. To enable paid answer generation, set `HOST_OPENAI_API_KEY` or `HOST_ANTHROPIC_API_KEY` on the server. The hosted configuration only forwards these explicitly named keys.

```sh
hostcompose() { docker compose --env-file .env.host -f compose.host.yaml "$@"; }
hostcompose build web
hostcompose run --rm init && hostcompose up -d --wait
hostcompose ps
```

The one-time setup builds both vector collections and the lexical indexes. Allow time for CPU embedding. After setup, Caddy obtains and renews the site's HTTPS certificate. Only Caddy publishes ports. Qdrant and the Python app communicate over the Compose network.

## Verify

Substitute your hostname in both commands. The readiness check requires populated vector collections and the lexical index files. The search also exercises the embedding model. The first query may need extra time to load a model, particularly if you select reranking.

```sh
curl --fail https://demo.yourdomain.com/ready
curl --fail https://demo.yourdomain.com/api/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"How do I thaw frozen breast milk?"}'
```

With empty provider keys, the JSON response includes source text and PDF page numbers in `retrieved_passages`. The public form and API share a limit of 20 accepted questions per minute across the single web process. Excess requests receive HTTP 429 with a retry delay. Hosted query logging is disabled. When you enable a key, questions and retrieved passages are sent to that provider.

## Operate

Run commands from the extracted directory. Define the `hostcompose` function again in a new shell. Use `hostcompose logs --tail 100 init web proxy qdrant` for service diagnostics. A one-off setup command prints progress and failures directly in the terminal.

- **Restart or change keys.** Run `hostcompose up -d --wait`. The indexes, model cache and certificates persist in named volumes.
- **Update the application.** Replace the application files, preserve `.env.host`, then run `hostcompose up -d --build --wait`.
- **Replace the corpus or change embedding settings.** Run `hostcompose stop proxy web` first. Then run `hostcompose run --rm --build init && hostcompose up -d --build --wait`. Setup replaces both vector collections. If a build stops partway through, repeat the setup command before bringing the site back online.
- **Stop the site.** Use `hostcompose down`. Adding `-v` deletes the saved data, model cache and certificates.

For a local HTTP smoke test, set `SITE_ADDRESS=http://localhost`, `HTTP_BIND=127.0.0.1:8080` and `HTTPS_BIND=127.0.0.1:8443` in `.env.host`. Follow the same setup and visit `http://localhost:8080`.
