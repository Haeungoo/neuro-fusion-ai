import {
  Activity,
  BarChart3,
  BrainCircuit,
  Database,
  Gauge,
  LineChart,
  ScanLine,
} from "lucide-react";

import AppShell from "@/components/AppShell";
import StatCard from "@/components/StatCard";
import StatusBadge from "@/components/StatusBadge";
import { apiFetch, mediaUrl } from "@/app/lib/api";

type FileInfo = {
  path?: string;
  exists: boolean;
  url?: string | null;
};

type SeizureMetrics = {
  accuracy?: number;
  precision?: number;
  recall?: number;
  sensitivity?: number;
  specificity?: number;
  f1_score?: number;
  balanced_accuracy?: number;

  true_positive?: number;
  true_negative?: number;
  false_positive?: number;
  false_negative?: number;

  num_samples?: number;
  num_windows?: number;
  num_files?: number;

  model?: string;
  dataset?: string;
};

type VisualizationMetadata = {
  source?: string;
  edf_path?: string;
  duration_seconds?: number;
  sampling_frequency?: number;
  channels?: string[];
  num_channels?: number;
  num_samples?: number;
  probability_threshold?: number;
  score_type?: string;
  note?: string;
};

type SeizureStatus = {
  module?: string;
  status?: string;
  description?: string;
  dataset_note?: string;
  labels?: Record<string, string>;
  pipeline?: string[];

  metrics?: SeizureMetrics | null;
  multi_file_metrics?: SeizureMetrics | null;

  visualization_metadata?: VisualizationMetadata | null;

  outputs?: {
    metrics_json?: FileInfo;
    predictions_csv?: FileInfo;
    confusion_matrix?: FileInfo;
    multi_file_metrics_json?: FileInfo;
    multi_file_confusion_matrix?: FileInfo;
    waveform?: FileInfo;
    probability_timeline?: FileInfo;
    visualization_metadata?: FileInfo;
    model?: FileInfo;
    model_file?: FileInfo;
  };

  visualization?: {
    available?: boolean;
    waveform?: FileInfo;
    probability_timeline?: FileInfo;
    metadata?: VisualizationMetadata | null;
  };

  disclaimer?: string;
};

export default async function SeizureDashboard() {
  const status = await apiFetch<SeizureStatus>("/api/seizure/status");

  const metrics =
    status?.multi_file_metrics ??
    status?.metrics ??
    null;

  const modelFile =
    status?.outputs?.model_file ??
    status?.outputs?.model;

  const confusionMatrixUrl =
    status?.outputs?.multi_file_confusion_matrix?.url
      ? mediaUrl(status.outputs.multi_file_confusion_matrix.url)
      : status?.outputs?.confusion_matrix?.url
        ? mediaUrl(status.outputs.confusion_matrix.url)
        : mediaUrl("seizure/seizure_confusion_matrix.png");

  const waveformUrl =
    status?.outputs?.waveform?.url
      ? mediaUrl(status.outputs.waveform.url)
      : mediaUrl("seizure/seizure_eeg_waveform.png");

  const probabilityTimelineUrl =
    status?.outputs?.probability_timeline?.url
      ? mediaUrl(status.outputs.probability_timeline.url)
      : mediaUrl("seizure/seizure_probability_timeline.png");

  const visualizationMetadata =
    status?.visualization_metadata ??
    status?.visualization?.metadata ??
    null;

  const visualizationAvailable =
    status?.visualization?.available ??
    Boolean(
      status?.outputs?.waveform?.exists &&
        status?.outputs?.probability_timeline?.exists,
    );

  return (
    <AppShell
      activePath="/seizure"
      title="EEG Seizure Detection"
      subtitle="CHB-MIT EEG analysis using Random Forest classification"
    >
      <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-6">
        <StatCard
          icon={Gauge}
          label="Accuracy"
          value={formatMetric(metrics?.accuracy)}
          description="Overall classification"
          iconClassName="bg-blue-50 text-blue-600"
        />

        <StatCard
          icon={BarChart3}
          label="F1 Score"
          value={formatMetric(metrics?.f1_score)}
          description="Seizure detection balance"
          iconClassName="bg-teal-50 text-teal-600"
        />

        <StatCard
          icon={Activity}
          label="Sensitivity"
          value={formatMetric(metrics?.sensitivity ?? metrics?.recall)}
          description="Seizure recall"
          iconClassName="bg-rose-50 text-rose-600"
        />

        <StatCard
          icon={ScanLine}
          label="Specificity"
          value={formatMetric(metrics?.specificity)}
          description="Non-seizure detection"
          iconClassName="bg-emerald-50 text-emerald-600"
        />

        <StatCard
          icon={Database}
          label="Files"
          value={formatCount(metrics?.num_files)}
          description="Evaluated EEG files"
          iconClassName="bg-cyan-50 text-cyan-600"
        />

        <StatCard
          icon={BrainCircuit}
          label="Status"
          value={status?.status ?? "N/A"}
          description="Backend output check"
          iconClassName="bg-amber-50 text-amber-600"
        />
      </section>

      <section className="mt-5 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h2 className="text-lg font-semibold text-slate-900">
              Seizure classification summary
            </h2>

            <p className="mt-1 text-sm text-slate-500">
              Random Forest classification results for seizure and non-seizure
              EEG segments.
            </p>
          </div>

          <StatusBadge active={modelFile?.exists ?? false} />
        </div>

        <div className="mt-5 grid gap-4 lg:grid-cols-2">
          <ImagePanel
            title="Confusion matrix"
            src={confusionMatrixUrl}
            compact
          />

          <InfoPanel
            title="Evaluation metrics"
            values={[
              ["Accuracy", formatMetric(metrics?.accuracy)],
              ["Precision", formatMetric(metrics?.precision)],
              [
                "Recall / Sensitivity",
                formatMetric(metrics?.recall ?? metrics?.sensitivity),
              ],
              ["Specificity", formatMetric(metrics?.specificity)],
              ["F1 score", formatMetric(metrics?.f1_score)],
              [
                "Balanced accuracy",
                formatMetric(metrics?.balanced_accuracy),
              ],
            ]}
          />
        </div>
      </section>

      <section className="mt-5 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h2 className="text-lg font-semibold text-slate-900">
              EEG Waveform and Seizure Probability Timeline
            </h2>

            <p className="mt-1 text-sm text-slate-500">
              The waveform shows a short EEG preview, while the probability
              timeline highlights seizure-like high-risk regions over time.
            </p>
          </div>

          <StatusBadge active={visualizationAvailable} />
        </div>

        <div className="mt-5 grid gap-4">
          <ImagePanel
            title="EEG waveform preview"
            src={waveformUrl}
          />

          <ImagePanel
            title="Seizure probability timeline"
            src={probabilityTimelineUrl}
            compact
          />
        </div>
      </section>

      <section className="mt-5 grid gap-5 lg:grid-cols-2">
        <InfoPanel
          title="Visualization details"
          values={[
            ["Source", visualizationMetadata?.source ?? "N/A"],
            [
              "Duration",
              formatSeconds(visualizationMetadata?.duration_seconds),
            ],
            [
              "Sampling frequency",
              formatHz(visualizationMetadata?.sampling_frequency),
            ],
            [
              "Channels",
              formatCount(visualizationMetadata?.num_channels),
            ],
            [
              "Threshold",
              formatMetric(visualizationMetadata?.probability_threshold),
            ],
            [
              "Score type",
              visualizationMetadata?.score_type ?? "N/A",
            ],
          ]}
        />

        <InfoPanel
          title="Classification counts"
          values={[
            ["True positive", formatCount(metrics?.true_positive)],
            ["True negative", formatCount(metrics?.true_negative)],
            ["False positive", formatCount(metrics?.false_positive)],
            ["False negative", formatCount(metrics?.false_negative)],
            [
              "Predictions CSV",
              status?.outputs?.predictions_csv?.exists
                ? "Saved"
                : "Missing",
            ],
            [
              "Model file",
              modelFile?.exists ? "Saved" : "Missing",
            ],
          ]}
        />
      </section>

      <section className="mt-5 grid gap-5 lg:grid-cols-2">
        <InfoPanel
          title="Model details"
          values={[
            ["Model", metrics?.model ?? "Random Forest"],
            ["Dataset", metrics?.dataset ?? "CHB-MIT"],
            ["Number of files", formatCount(metrics?.num_files)],
            ["Number of windows", formatCount(metrics?.num_windows)],
            ["Number of samples", formatCount(metrics?.num_samples)],
            [
              "Metrics JSON",
              status?.outputs?.metrics_json?.exists ? "Saved" : "Missing",
            ],
          ]}
        />

        <InfoPanel
          title="Output files"
          values={[
            [
              "Confusion matrix",
              status?.outputs?.confusion_matrix?.exists ||
              status?.outputs?.multi_file_confusion_matrix?.exists
                ? "Saved"
                : "Missing",
            ],
            [
              "Waveform image",
              status?.outputs?.waveform?.exists ? "Saved" : "Missing",
            ],
            [
              "Probability timeline",
              status?.outputs?.probability_timeline?.exists
                ? "Saved"
                : "Missing",
            ],
            [
              "Visualization metadata",
              status?.outputs?.visualization_metadata?.exists
                ? "Saved"
                : "Missing",
            ],
            [
              "Predictions CSV",
              status?.outputs?.predictions_csv?.exists
                ? "Saved"
                : "Missing",
            ],
            [
              "Model file",
              modelFile?.exists ? "Saved" : "Missing",
            ],
          ]}
        />
      </section>

      <section className="mt-5 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
        <h2 className="text-lg font-semibold text-slate-900">
          Seizure detection pipeline
        </h2>

        <p className="mt-1 text-sm text-slate-500">
          The pipeline converts raw EEG recordings into window-level features
          and model predictions.
        </p>

        <div className="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6">
          {(status?.pipeline ?? [
            "EEG recording",
            "Signal preprocessing",
            "Window segmentation",
            "Feature extraction",
            "Random Forest classification",
            "Seizure probability visualization",
          ]).map((step, index) => (
            <PipelineStep
              key={step}
              index={index}
              label={step}
            />
          ))}
        </div>
      </section>

      <section className="mt-5 rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-800">
        {status?.disclaimer ??
          "This model is for research and educational use only and is not intended for clinical diagnosis."}
      </section>
    </AppShell>
  );
}

function ImagePanel({
  title,
  src,
  compact = false,
}: {
  title: string;
  src: string;
  compact?: boolean;
}) {
  return (
    <figure className="overflow-hidden rounded-xl border border-slate-200">
      <div
        className={[
          "flex items-center justify-center bg-slate-950 p-3",
          compact ? "h-72" : "h-96",
        ].join(" ")}
      >
        <img
          src={src}
          alt={title}
          className="max-h-full max-w-full object-contain"
        />
      </div>

      <figcaption className="bg-white px-4 py-3 text-center text-sm font-medium text-slate-700">
        {title}
      </figcaption>
    </figure>
  );
}

function InfoPanel({
  title,
  values,
}: {
  title: string;
  values: Array<[string, string]>;
}) {
  return (
    <article className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <h2 className="text-lg font-semibold text-slate-900">
        {title}
      </h2>

      <dl className="mt-4 divide-y divide-slate-100">
        {values.map(([label, value]) => (
          <MetricRow
            key={label}
            label={label}
            value={value}
          />
        ))}
      </dl>
    </article>
  );
}

function MetricRow({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="flex items-center justify-between gap-4 py-3">
      <dt className="text-sm text-slate-500">{label}</dt>

      <dd className="text-right text-sm font-semibold text-slate-800">
        {value}
      </dd>
    </div>
  );
}

function PipelineStep({
  index,
  label,
}: {
  index: number;
  label: string;
}) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4">
      <p className="text-xs font-semibold text-blue-600">
        {`0${index + 1}`}
      </p>

      <p className="mt-2 text-sm font-semibold text-slate-800">
        {label}
      </p>
    </div>
  );
}

function formatMetric(value?: number): string {
  return value == null ? "N/A" : value.toFixed(4);
}

function formatCount(value?: number): string {
  return value == null ? "N/A" : value.toLocaleString();
}

function formatHz(value?: number): string {
  return value == null ? "N/A" : `${value.toFixed(1)} Hz`;
}

function formatSeconds(value?: number): string {
  return value == null ? "N/A" : `${value.toFixed(1)} s`;
}