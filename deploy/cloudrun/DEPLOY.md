# Deploying the assistant to Google Cloud Run

Everything below is one-time except the last section. Set these once per shell:

```bash
export PROJECT=<your-project-id>
export REGION=europe-west9                  # Paris — closest to a French client
export IMAGE=$REGION-docker.pkg.dev/$PROJECT/apps/rag:v1
```

## 1. Project and APIs

Cloud Run needs a billing account attached to the project. The free tier covers
a demo comfortably, but the card has to be on file before anything runs.

```bash
gcloud auth login
gcloud config set project $PROJECT
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
                       artifactregistry.googleapis.com secretmanager.googleapis.com
gcloud artifacts repositories create apps \
  --repository-format=docker --location=$REGION
```

## 2. The API key as a secret

Not an environment variable on the service: a plain env var is readable by
anyone with Viewer on the project and it ends up in deployment logs and in
`gcloud run services describe` output.

```bash
printf '%s' 'gsk_...' | gcloud secrets create groq-api-key --data-file=-

# The Cloud Run service identity has to be allowed to read it, which is not
# implied by creating it.
PROJECT_NUMBER=$(gcloud projects describe $PROJECT --format='value(projectNumber)')
gcloud secrets add-iam-policy-binding groq-api-key \
  --member="serviceAccount:$PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
  --role=roles/secretmanager.secretAccessor
```

## 3. Build

Roughly fifteen minutes: it downloads the corpus, indexes 2 634 extracts and
bakes both models in. Cloud Build, not the laptop — the image is ~2.5 GB and
pushing that over a home upload link is the slow way round.

```bash
gcloud builds submit --config deploy/cloudrun/cloudbuild.yaml \
  --substitutions=_IMAGE=$IMAGE
```

Watch for `indexed 2634 chunks` in the log. If that line is missing the build
failed at the step that matters, whatever else it printed.

## 4. Deploy

```bash
gcloud run deploy assistant-droit-travail \
  --image=$IMAGE \
  --region=$REGION \
  --allow-unauthenticated \
  --memory=2Gi \
  --cpu=2 \
  --concurrency=4 \
  --max-instances=3 \
  --timeout=120 \
  --set-secrets=GROQ_API_KEY=groq-api-key:latest
```

Why these numbers:

- **2 GiB** — torch, the embedder, the reranker and the Chroma index together
  sit around 1.2 GB resident. 1 GiB dies during warm-up, and Cloud Run reports
  it as a generic startup failure.
- **concurrency 4** — the retrieval path is synchronous CPU work. Letting the
  default 80 requests onto one instance means everybody waits behind everybody.
- **max-instances 3** — a ceiling on both the bill and the Groq quota. There is
  no scenario where a portfolio demo legitimately needs a fourth instance, and
  plenty where a crawler asks for forty.

The command prints the service URL. That is the address for the README.

## 5. The cold start, and what to do about it

With no traffic Cloud Run scales to zero. The next visitor waits while the
image starts and the models load — around a minute. For a link in a cold email
that is bad enough to matter.

Keeping an instance always warm is a paid setting (`--min-instances=1` bills
for a container that sits idle around the clock). The cheap version is a ping:

```bash
gcloud scheduler jobs create http rag-warm \
  --location=$REGION \
  --schedule='*/5 * * * *' \
  --uri="<service-url>/health" \
  --http-method=GET
```

Cloud Run bills request-processing time, and `/health` is milliseconds of it,
so this stays inside the free tier while keeping an instance alive through the
working day. Three scheduler jobs are free.

If you would rather not run it permanently: `gcloud scheduler jobs resume
rag-warm` the morning you send the emails, `pause` when the week is over.

## 6. Updating

```bash
gcloud builds submit --config deploy/cloudrun/cloudbuild.yaml \
  --substitutions=_IMAGE=$REGION-docker.pkg.dev/$PROJECT/apps/rag:v2
gcloud run deploy assistant-droit-travail --image=$REGION-docker.pkg.dev/$PROJECT/apps/rag:v2 --region=$REGION
```

A new tag each time, not `:latest`. Cloud Run keeps revisions, so a bad deploy
is one `gcloud run services update-traffic --to-revisions=<previous>=100` away
from being undone — but only if the previous image still exists under its own
tag.

## 7. Watch the first week

```bash
gcloud run services logs read assistant-droit-travail --region=$REGION --limit=50
```

Two things worth seeing early: `429 rate limited` from Groq (the free quota is
being consumed, by visitors or by a crawler), and `refused:` lines (the
guardrail declining questions — a few are healthy, a wall of them means
something is wrong with retrieval).
