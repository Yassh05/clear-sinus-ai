import { useState } from "react";
import { supabase } from "@/integrations/supabase/client";
import { SymptomForm, type SymptomFormData } from "@/components/SymptomForm";
import { PredictionResult } from "@/components/PredictionResult";
import { FollowUpChat } from "@/components/FollowUpChat";
import { toast } from "sonner";
import { Activity, Brain, ShieldCheck, Database, GitCompare } from "lucide-react";
import { ThemeToggle } from "@/components/ThemeToggle";

const Index = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [lastFormData, setLastFormData] = useState<SymptomFormData | null>(null);

  const handleSubmit = async (data: SymptomFormData) => {
    setIsLoading(true);
    setLastFormData(data);
    try {
      const { data: prediction, error } = await supabase.functions.invoke("sinus-predict", {
        body: data,
      });

      if (error) {
        if (error.message?.includes("429")) {
          toast.error("Rate limit exceeded. Please try again in a moment.");
        } else if (error.message?.includes("402")) {
          toast.error("Service credits exhausted. Please try again later.");
        } else {
          toast.error("Failed to analyze symptoms. Please try again.");
        }
        return;
      }

      setResult(prediction);
    } catch (err) {
      toast.error("An unexpected error occurred.");
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-surface-elevated/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="container max-w-4xl mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-primary/10 flex items-center justify-center">
              <Activity className="w-5 h-5 text-primary" />
            </div>
            <div>
              <h1 className="font-display text-base font-bold text-foreground leading-tight">SinusAI</h1>
              <p className="text-[11px] text-muted-foreground leading-none">ML-Powered Disease Prediction</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
              <ShieldCheck className="w-3.5 h-3.5 text-success" />
              <span>XGBoost + AI</span>
            </div>
            <ThemeToggle />
          </div>
        </div>
      </header>

      <main className="container max-w-4xl mx-auto px-4 py-8 pb-20">
        {!result ? (
          <>
            {/* Hero */}
            <section className="text-center mb-10 opacity-0 animate-fade-up">
              <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-primary/10 text-primary text-xs font-medium mb-4">
                <Brain className="w-3.5 h-3.5" />
                Machine Learning Pipeline
              </div>
              <h2 className="font-display text-3xl sm:text-4xl font-bold text-foreground mb-3 leading-[1.1] tracking-tight">
                AI-Based Sinus & Nasal<br />Disease Prediction System
              </h2>
              <p className="text-muted-foreground max-w-lg mx-auto leading-relaxed">
                Powered by XGBoost, Random Forest, Logistic Regression & Decision Trees
                with AI-driven clinical validation and explainable predictions.
              </p>
            </section>

            {/* Features */}
            <section className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-10">
              {[
                { icon: Brain, label: "14+ Symptoms", desc: "Comprehensive input" },
                { icon: GitCompare, label: "4 ML Models", desc: "Ensemble comparison" },
                { icon: Database, label: "Feature Eng.", desc: "Weighted scoring" },
                { icon: ShieldCheck, label: "Explainable AI", desc: "Transparent results" },
              ].map((f, i) => (
                <div
                  key={f.label}
                  className="text-center p-4 rounded-xl bg-card border border-border opacity-0 animate-fade-up"
                  style={{ animationDelay: `${i * 80 + 200}ms` }}
                >
                  <f.icon className="w-5 h-5 text-primary mx-auto mb-1.5" />
                  <p className="font-display text-sm font-semibold text-foreground">{f.label}</p>
                  <p className="text-xs text-muted-foreground mt-0.5">{f.desc}</p>
                </div>
              ))}
            </section>

            <div className="bg-card rounded-2xl border border-border p-6 sm:p-8 shadow-sm">
              <SymptomForm onSubmit={handleSubmit} isLoading={isLoading} />
            </div>
          </>
        ) : (
          <>
            <div className="max-w-2xl mx-auto">
              <PredictionResult result={result} onReset={() => setResult(null)} />
            </div>
            <FollowUpChat predictionContext={result} />
          </>
        )}
      </main>
    </div>
  );
};

export default Index;
