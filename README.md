# Clear-Sinus AI

Clear-Sinus AI is a full-stack medical support application for sinus and nasal condition assessment.
Users provide symptoms and context, the app runs a trained ML inference pipeline via Supabase Edge Functions, enriches outputs with AI clinical reasoning, and returns a clear report with follow-up chat.

## Highlights

- Guided symptom intake with 14+ symptoms and risk-factor inputs
- Optional upload of medical reports (PDF/JPG/PNG/WEBP)
- ML-driven prediction pipeline with model comparison and explainability
- AI clinical reasoning layer with recommendations and differential diagnoses
- Probability distribution and feature-importance visualization
- Follow-up conversational assistant based on prediction context
- Downloadable PDF report for each assessment
- Responsive React UI with light/dark theme support

## Tech Stack

- Frontend: React 18, TypeScript, Vite, Tailwind CSS, shadcn/ui, Radix UI
- Data/UI utilities: TanStack Query, Recharts, React Hook Form, Sonner
- Backend: Supabase (Edge Functions + Storage)
- AI Gateway: model calls from Supabase Edge Functions
- Testing: Vitest

## Project Structure

```text
clear-sinus-ai/
	src/
		components/
			SymptomForm.tsx
			PredictionResult.tsx
			FollowUpChat.tsx
		pages/
			Index.tsx
		integrations/supabase/
			client.ts
		utils/
			generateReport.ts
	supabase/
		functions/
			sinus-predict/
			sinus-chat/
		migrations/
	public/
	index.html
```

## How It Works

1. User submits symptoms and profile inputs from the frontend form.
2. Frontend calls Supabase Edge Function `sinus-predict`.
3. `sinus-predict`:
	 - builds feature vectors,
	 - runs logistic-regression style inference using embedded trained parameters,
	 - computes probability distribution and feature importance,
	 - optionally reads uploaded report files,
	 - enriches output using an AI model for clinical reasoning.
4. Frontend renders:
	 - risk level,
	 - confidence,
	 - model comparison,
	 - explainability charts,
	 - recommendations and doctor-escalation guidance.
5. Follow-up chat calls `sinus-chat` with prediction context for contextual Q&A.

## Prerequisites

- Node.js 18+
- npm 9+
- A Supabase project
- Supabase CLI (recommended for local function/deployment workflows)

## Environment Variables

Create a `.env` file in the project root.

Required frontend vars:

```env
VITE_SUPABASE_URL=your_supabase_project_url
VITE_SUPABASE_PUBLISHABLE_KEY=your_supabase_anon_key
VITE_SUPABASE_PROJECT_ID=your_supabase_project_id
```

Required server-side secret for Edge Functions:

```bash
LOVABLE_API_KEY=your_ai_gateway_key
```

Note: the Edge Function code currently expects the secret name `LOVABLE_API_KEY`.

## Local Development

Install dependencies:

```bash
npm install
```

Start frontend dev server:

```bash
npm run dev
```

Build production bundle:

```bash
npm run build
```

Preview production build:

```bash
npm run preview
```

Run tests:

```bash
npm run test
```

## Supabase Setup

1. Link your Supabase project.
2. Apply migration(s) under `supabase/migrations`.
3. Deploy Edge Functions:

```bash
supabase functions deploy sinus-predict
supabase functions deploy sinus-chat
```

4. Set function secret(s):

```bash
supabase secrets set LOVABLE_API_KEY=your_key
```

5. Ensure storage bucket exists:
	 - Bucket name: `medical-reports`
	 - Current migration config is public-read/public-upload for anon/authenticated.

## NPM Scripts

- `npm run dev`: start Vite development server
- `npm run build`: production build
- `npm run build:dev`: build in development mode
- `npm run lint`: run ESLint
- `npm run preview`: preview built app
- `npm run test`: run Vitest tests
- `npm run test:watch`: watch mode tests

## Security Notes

- Uploaded medical files are currently stored in a public bucket per migration policy.
- For production healthcare workflows, tighten storage and RLS policies.
- Avoid committing real secrets in `.env`.
- This tool is informational and not a replacement for clinician diagnosis.

## Disclaimer

This application is for informational and educational use only.
It does not provide medical diagnosis, treatment, or emergency care.
Users should consult qualified healthcare professionals for clinical decisions.

## License

Add your preferred license in this repository (for example MIT, Apache-2.0, or proprietary internal use).
