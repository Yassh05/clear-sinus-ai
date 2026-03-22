import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers":
    "authorization, x-client-info, apikey, content-type, x-supabase-client-platform, x-supabase-client-platform-version, x-supabase-client-runtime, x-supabase-client-runtime-version",
};

serve(async (req) => {
  if (req.method === "OPTIONS") return new Response(null, { headers: corsHeaders });

  try {
    const { symptoms, duration, severity, age, allergies } = await req.json();

    const LOVABLE_API_KEY = Deno.env.get("LOVABLE_API_KEY");
    if (!LOVABLE_API_KEY) throw new Error("LOVABLE_API_KEY is not configured");

    const symptomLabels: Record<string, string> = {
      nasal_congestion: "Nasal Congestion/Blockage",
      facial_pain: "Facial Pain/Pressure",
      nasal_discharge: "Thick Nasal Discharge",
      reduced_smell: "Reduced Sense of Smell",
      headache: "Headache",
      post_nasal_drip: "Post-Nasal Drip",
      cough: "Cough",
      ear_pressure: "Ear Pressure/Fullness",
      fever: "Fever",
      fatigue: "Fatigue/Malaise",
      dental_pain: "Upper Dental Pain",
      bad_breath: "Bad Breath",
      snoring: "Snoring/Sleep Issues",
      nosebleeds: "Nosebleeds",
    };

    const symptomNames = symptoms.map((s: string) => symptomLabels[s] || s).join(", ");

    const prompt = `You are a medical AI prediction system specialized in sinus and nasal diseases. Based on the following patient data, provide a structured prediction.

Patient Data:
- Age: ${age}
- Symptoms: ${symptomNames}
- Duration: ${duration.replace(/_/g, " ")}
- Severity: ${severity}
- Known allergies: ${allergies ? "Yes" : "No"}

Analyze these symptoms and provide a prediction using your medical knowledge. Consider conditions like: Acute Sinusitis, Chronic Sinusitis, Allergic Rhinitis, Non-Allergic Rhinitis, Nasal Polyps, Deviated Septum, Rhinosinusitis, Fungal Sinusitis, and others.

Return a JSON prediction with appropriate confidence levels and risk assessment.`;

    const response = await fetch("https://ai.gateway.lovable.dev/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${LOVABLE_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "google/gemini-3-flash-preview",
        messages: [
          {
            role: "system",
            content:
              "You are a medical AI system. Always respond with accurate, evidence-based assessments. Be clear about limitations.",
          },
          { role: "user", content: prompt },
        ],
        tools: [
          {
            type: "function",
            function: {
              name: "sinus_prediction",
              description: "Return a structured sinus/nasal disease prediction",
              parameters: {
                type: "object",
                properties: {
                  condition: {
                    type: "string",
                    description: "Primary predicted condition name",
                  },
                  confidence: {
                    type: "number",
                    description: "Confidence percentage 0-100",
                  },
                  risk_level: {
                    type: "string",
                    enum: ["low", "moderate", "high"],
                  },
                  description: {
                    type: "string",
                    description: "Brief description of the predicted condition",
                  },
                  recommendations: {
                    type: "array",
                    items: { type: "string" },
                    description: "3-5 treatment/management recommendations",
                  },
                  when_to_see_doctor: {
                    type: "string",
                    description: "Guidance on when to seek professional medical help",
                  },
                  differential_diagnoses: {
                    type: "array",
                    items: {
                      type: "object",
                      properties: {
                        name: { type: "string" },
                        likelihood: {
                          type: "string",
                          enum: ["Low", "Moderate", "High"],
                        },
                      },
                      required: ["name", "likelihood"],
                    },
                    description: "2-4 alternative possible conditions",
                  },
                },
                required: [
                  "condition",
                  "confidence",
                  "risk_level",
                  "description",
                  "recommendations",
                  "when_to_see_doctor",
                  "differential_diagnoses",
                ],
                additionalProperties: false,
              },
            },
          },
        ],
        tool_choice: { type: "function", function: { name: "sinus_prediction" } },
      }),
    });

    if (!response.ok) {
      const status = response.status;
      const text = await response.text();
      console.error("AI gateway error:", status, text);
      return new Response(JSON.stringify({ error: "AI analysis failed" }), {
        status,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    const aiData = await response.json();
    const toolCall = aiData.choices?.[0]?.message?.tool_calls?.[0];

    if (!toolCall) {
      throw new Error("No prediction returned from AI model");
    }

    const prediction = JSON.parse(toolCall.function.arguments);

    return new Response(JSON.stringify(prediction), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  } catch (e) {
    console.error("sinus-predict error:", e);
    return new Response(
      JSON.stringify({ error: e instanceof Error ? e.message : "Unknown error" }),
      {
        status: 500,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      }
    );
  }
});
