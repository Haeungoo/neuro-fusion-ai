"use client";

import { useEffect, useMemo, useState } from "react";
import { apiFetch, mediaUrl } from "@/app/lib/api";

type ModeKey =
  | "synthetic"
  | "chbmit_one_file"
  | "chbmit_multi_file";

type FileInfo = {
  exists: boolean;
  path?: string;
  url: string | null;
};

type SeizureEvaluationMetrics = {
  classification?: {
    true_positive?: number;
    true_negative?: number;
    false_positive?: number;
    false_negative?: number;
    sensitivity?: number;
    recall?: number;
    specificity?: number;
    precision?: number;
    negative_predictive_value?: number;
    f1_score?: number;
    accuracy?: number;
    balanced_accuracy?: number;
    prevalence?: number;
    false_positive_rate?: number;
    false_negative_rate?: number;
    num_samples?: number;
    num_positive_samples?: number;
    num_negative_samples?: number;
  };
  false_alarm_analysis?: {
    false_positive_windows?: number;
    false_alarm_events?: number;
    non_seizure_windows?: number;
    non_seizure_duration_seconds?: number;
    non_seizure_duration_hours?: number;
    false_alarms_per_hour?: number;
    merge_consecutive_windows?: boolean;
    step_seconds?: number;
  };
};

type SeizureStatusResponse = {
  module?: string;
  status?: string;
  description?: string;
  available_modes?: string[];
  results?: {
    synthetic?: {
      waveform?: FileInfo;
      probability_timeline?: FileInfo;
      confusion_matrix?: FileInfo;
    };
    chbmit_one_file?: {
      waveform?: FileInfo;
      probability_timeline?: FileInfo;
      confusion_matrix?: FileInfo;
    };
    chbmit_multi_file?: {
      waveform?: FileInfo;
      probability_timeline?: FileInfo;
      confusion_matrix?: FileInfo;
    };
  };
  training_outputs?: {
    feature_dataset?: FileInfo;
    prediction_csv?: FileInfo;
    random_forest_metrics?: Record<string, unknown> | null;
    random_forest_confusion_matrix?: FileInfo;
  };
  evaluation_metrics?: SeizureEvaluationMetrics | null;
  evaluation_outputs?: {
    per_file_metrics_csv?: FileInfo;
    confusion_matrix?: FileInfo;
  };
};

const MODE_LABELS: Record<ModeKey, string> = {
  synthetic: "Synthetic demo",
  chbmit_one_file: "CHB-MIT one file",
  chbmit_multi_file: "CHB-MIT multi-file",
};

const MODE_DESCRIPTIONS: Record<ModeKey, string> = {
  synthetic:
    "A simple artificial EEG example used only for visual demonstration.",
  chbmit_one_file:
    "A single CHB-MIT EEG file result. Useful for understanding one recording.",
  chbmit_multi_file:
    "A multi-file CHB-MIT result. Useful for a broader demo visualization.",
};

const RESULT_FILENAMES: Record<
  ModeKey,
  {
    waveform: string;
    timeline: string;
    confusionMatrix: string;
  }
> = {
  synthetic: {
    waveform: "seizure/synthetic_waveform.png",
    timeline: "seizure/synthetic_probability_timeline.png",
    confusionMatrix: "seizure/synthetic_confusion_matrix.png",
  },
  chbmit_one_file: {
    waveform: "seizure/chbmit_chb01_03_waveform.png",
    timeline: "seizure/chbmit_chb01_03_probability_timeline.png",
    confusionMatrix: "seizure/chbmit_chb01_03_confusion_matrix.png",
  },
  chbmit_multi_file: {
    waveform: "seizure/chbmit_multi_file_waveform.png",
    timeline: "seizure/chbmit_multi_file_probability_timeline.png",
    confusionMatrix: "seizure/chbmit_multi_file_confusion_matrix.png",
  },
};

function formatPercent(value?: number): string {
  if (value === undefined || value === null) {
    return "N/A";
  }

  return `${(value * 100).toFixed(1)}%`;
}

function formatNumber(
  value?: number,
  digits: number = 2,
): string {
  if (value === undefined || value === null) {
    return "N/A";
  }

  return value.toFixed(digits);
}

function backendMediaUrl(url?: string | null): string | null {
  if (!url) {
    return null;
  }

  if (url.startsWith("http://") || url.startsWith("https://")) {
    return url;
  }

  return mediaUrl(url);
}

function MetricCard({
  label,
  value,
  description,
  emphasis = false,
}: {
  label: string;
  value: string;
  description: string;
  emphasis?: boolean;
}) {
  return (
    <div
      className={[
        "rounded-2xl border p-4 shadow-sm",
        emphasis
          ? "border-blue-200 bg-blue-50"
          : "border-slate-200 bg-white",
      ].join(" ")}
    >
      <p className="text-sm text-slate-500">{label}</p>

      <p
        className={[
          "mt-2 text-2xl font-semibold",
          emphasis ? "text-blue-700" : "text-slate-900",
        ].join(" ")}
      >
        {value}
      </p>

      <p className="mt-2 text-xs leading-5 text-slate-500">
        {description}
      </p>
    </div>
  );
}

function ImageCard({
  title,
  description,
  src,
  alt,
}: {
  title: string;
  description: string;
  src: string;
  alt: string;
}) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="mb-3">
        <h3 className="text-base font-semibold text-slate-900">
          {title}
        </h3>

        <p className="mt-1 text-sm leading-5 text-slate-500">
          {description}
        </p>
      </div>

      <img
        src={src}
        alt={alt}
        className="w-full rounded-xl border border-slate-100"
      />
    </div>
  );
}

function SectionHeader({
  label,
  title,
  description,
}: {
  label: string;
  title: string;
  description: string;
}) {
  return (
    <div className="mb-5">
      <p className="text-sm font-medium uppercase tracking-wide text-blue-600">
        {label}
      </p>

      <h2 className="mt-1 text-2xl font-semibold text-slate-950">
        {title}
      </h2>

      <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-500">
        {description}
      </p>
    </div>
  );
}

export default function SeizureDashboard() {
  const [selectedMode, setSelectedMode] =
    useState<ModeKey>("chbmit_multi_file");

  const [status, setStatus] =
    useState<SeizureStatusResponse | null>(null);

  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadStatus() {
      setLoading(true);

      const data = await apiFetch<SeizureStatusResponse>(
        "/api/seizure/status",
      );

      setStatus(data);
      setLoading(false);
    }

    loadStatus();
  }, []);

  const selectedResult = RESULT_FILENAMES[selectedMode];

  const evaluationMetrics = status?.evaluation_metrics;
  const classification = evaluationMetrics?.classification;
  const falseAlarm =
    evaluationMetrics?.false_alarm_analysis;

  const evaluationConfusionMatrixUrl = useMemo(() => {
    return backendMediaUrl(
      status?.evaluation_outputs?.confusion_matrix?.url,
    );
  }, [status]);

  return (
    <main className="min-h-screen bg-slate-50 px-6 py-8 text-slate-900">
      <div className="mx-auto max-w-7xl">
        <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
          <div className="flex flex-col gap-5 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <p className="text-sm font-medium uppercase tracking-wide text-blue-600">
                NeuroFusion-AI
              </p>

              <h1 className="mt-2 text-3xl font-bold text-slate-950">
                EEG Seizure Detection
              </h1>

              <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-600">
                This page separates the final model evaluation
                from the demo visualizations. The top section
                shows the true model performance. The lower
                section lets you explore example EEG outputs.
              </p>
            </div>

            <div className="rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3">
              <p className="text-sm text-slate-500">
                Backend status
              </p>

              <p className="mt-1 text-lg font-semibold text-slate-900">
                {loading
                  ? "Loading..."
                  : status?.status ?? "Unavailable"}
              </p>
            </div>
          </div>
        </section>

        <section className="mt-8 rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
          <SectionHeader
            label="Step 1"
            title="Final Model Evaluation"
            description="These numbers come from the final evaluation pipeline, using results/seizure/seizure_metrics.json. This is the most important section for judging model performance."
          />

          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            <MetricCard
              label="Sensitivity"
              value={formatPercent(
                classification?.sensitivity,
              )}
              description="How well the model detects true seizure windows. This is clinically important because missed seizures are dangerous."
              emphasis
            />

            <MetricCard
              label="Specificity"
              value={formatPercent(
                classification?.specificity,
              )}
              description="How well the model correctly identifies non-seizure windows."
              emphasis
            />

            <MetricCard
              label="Precision"
              value={formatPercent(
                classification?.precision,
              )}
              description="Among predicted seizure windows, how many were truly seizure."
            />

            <MetricCard
              label="F1 score"
              value={formatPercent(
                classification?.f1_score,
              )}
              description="A balance between sensitivity and precision."
            />

            <MetricCard
              label="Accuracy"
              value={formatPercent(
                classification?.accuracy,
              )}
              description="Overall proportion of correctly classified windows. In seizure data, this should not be interpreted alone."
            />

            <MetricCard
              label="False alarms/hour"
              value={formatNumber(
                falseAlarm?.false_alarms_per_hour,
                2,
              )}
              description="Estimated false seizure alarm events per non-seizure hour."
              emphasis
            />
          </div>

          {evaluationConfusionMatrixUrl && (
            <div className="mt-8">
              <ImageCard
                title="Final Evaluation Confusion Matrix"
                description="This confusion matrix belongs to the final evaluation result, not just one demo image mode."
                src={evaluationConfusionMatrixUrl}
                alt="Final seizure evaluation confusion matrix"
              />
            </div>
          )}
        </section>

        <section className="mt-8 rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
          <SectionHeader
            label="Step 2"
            title="Demo Result Viewer"
            description="This section is for visual exploration only. Changing the mode below changes the example images, but it does not change the final evaluation metrics above."
          />

          <div className="grid gap-4 md:grid-cols-3">
            {(
              Object.keys(MODE_LABELS) as ModeKey[]
            ).map((mode) => {
              const isSelected = selectedMode === mode;

              return (
                <button
                  key={mode}
                  type="button"
                  onClick={() => setSelectedMode(mode)}
                  className={[
                    "rounded-2xl border p-4 text-left shadow-sm transition",
                    isSelected
                      ? "border-blue-500 bg-blue-50"
                      : "border-slate-200 bg-white hover:border-blue-300",
                  ].join(" ")}
                >
                  <p className="text-base font-semibold text-slate-900">
                    {MODE_LABELS[mode]}
                  </p>

                  <p className="mt-2 text-sm leading-5 text-slate-500">
                    {MODE_DESCRIPTIONS[mode]}
                  </p>

                  {isSelected && (
                    <p className="mt-3 text-xs font-medium text-blue-700">
                      Selected demo mode
                    </p>
                  )}
                </button>
              );
            })}
          </div>

          <div className="mt-8 grid gap-6 lg:grid-cols-2">
            <ImageCard
              title="EEG Waveform"
              description="This shows a representative EEG signal segment for the selected demo mode."
              src={mediaUrl(selectedResult.waveform)}
              alt="EEG waveform"
            />

            <ImageCard
              title="Seizure Probability Timeline"
              description="This shows the predicted seizure probability across time windows for the selected demo mode."
              src={mediaUrl(selectedResult.timeline)}
              alt="Seizure probability timeline"
            />

            <ImageCard
              title="Demo Confusion Matrix"
              description="This confusion matrix belongs to the selected demo mode. It is separate from the final evaluation confusion matrix above."
              src={mediaUrl(selectedResult.confusionMatrix)}
              alt="Demo seizure confusion matrix"
            />
          </div>
        </section>

        <section className="mt-8 rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
          <SectionHeader
            label="Step 3"
            title="Technical Details"
            description="These values are useful for debugging and understanding how the final metrics were calculated."
          />

          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            <MetricCard
              label="True positive"
              value={String(
                classification?.true_positive ?? "N/A",
              )}
              description="Seizure windows correctly detected."
            />

            <MetricCard
              label="True negative"
              value={String(
                classification?.true_negative ?? "N/A",
              )}
              description="Non-seizure windows correctly classified."
            />

            <MetricCard
              label="False positive"
              value={String(
                classification?.false_positive ?? "N/A",
              )}
              description="Non-seizure windows incorrectly flagged as seizure."
            />

            <MetricCard
              label="False negative"
              value={String(
                classification?.false_negative ?? "N/A",
              )}
              description="Seizure windows missed by the model."
            />

            <MetricCard
              label="Total samples"
              value={String(
                classification?.num_samples ?? "N/A",
              )}
              description="Number of evaluated EEG windows."
            />

            <MetricCard
              label="Seizure samples"
              value={String(
                classification?.num_positive_samples ?? "N/A",
              )}
              description="Number of true seizure windows."
            />

            <MetricCard
              label="False alarm events"
              value={String(
                falseAlarm?.false_alarm_events ?? "N/A",
              )}
              description="Consecutive false-positive windows merged into alarm events."
            />

            <MetricCard
              label="Step seconds"
              value={formatNumber(
                falseAlarm?.step_seconds,
                1,
              )}
              description="Time interval between adjacent EEG windows."
            />
          </div>
        </section>
      </div>
    </main>
  );
}