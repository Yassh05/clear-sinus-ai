import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers":
    "authorization, x-client-info, apikey, content-type, x-supabase-client-platform, x-supabase-client-platform-version, x-supabase-client-runtime, x-supabase-client-runtime-version",
};

serve(async (req) => {
  if (req.method === "OPTIONS") return new Response(null, { headers: corsHeaders });

  try {
    const { messages, predictionContext } = await req.json();

    const LOVABLE_API_KEY = Deno.env.get("LOVABLE_API_KEY");
    if (!LOVABLE_API_KEY) throw new Error("LOVABLE_API_KEY is not configured");

    const systemPrompt = `You are a helpful, empathetic medical AI assistant for SinusAI, a sinus and nasal disease prediction system. A patient has just received an ML-based prediction and wants to ask follow-up questions.

Here is the prediction context:
- Predicted Condition: ${predictionContext.condition}
- Confidence: ${predictionContext.confidence}%
- Risk Level: ${predictionContext.risk_level}
- Description: ${predictionContext.description}
- Clinical Reasoning: ${predictionContext.clinical_reasoning}
- Recommendations: ${predictionContext.recommendations?.join(", ")}
- When to See Doctor: ${predictionContext.when_to_see_doctor}
- Differential Diagnoses: ${predictionContext.differential_diagnoses?.map((d: any) => `${d.name} (${d.likelihood})`).join(", ")}
${predictionContext.report_findings ? `- Report Findings: ${predictionContext.report_findings}` : ""}

Guidelines:
- Answer questions about the predicted condition, symptoms, treatments, and lifestyle changes.
- Be clear, concise, and use simple language. Use markdown formatting.
- Always remind users that this is informational only and not a substitute for professional medical advice.
- If asked about medications or specific treatments, provide general information but emphasize consulting a doctor.
- Stay within the scope of sinus/nasal conditions. Politely redirect off-topic medical questions.
- Be warm and supportive. Patients may be anxious about their results.`;

    const response = await fetch("https://ai.gateway.lovable.dev/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${LOVABLE_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "google/gemini-3-flash-preview",
        messages: [
          { role: "system", content: systemPrompt },
          ...messages,
        ],
        stream: true,
      }),
    });

    if (!response.ok) {
      if (response.status === 429) {
        return new Response(JSON.stringify({ error: "Rate limit exceeded. Please try again in a moment." }), {
          status: 429,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        });
      }
      if (response.status === 402) {
        return new Response(JSON.stringify({ error: "Service credits exhausted. Please try again later." }), {
          status: 402,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        });
      }
      const text = await response.text();
      console.error("AI gateway error:", response.status, text);
      return new Response(JSON.stringify({ error: "AI chat failed" }), {
        status: 500,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    return new Response(response.body, {
      headers: { ...corsHeaders, "Content-Type": "text/event-stream" },
    });
  } catch (e) {
    console.error("sinus-chat error:", e);
    return new Response(
      JSON.stringify({ error: e instanceof Error ? e.message : "Unknown error" }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  }
});
