"use client";

import {
  Activity,
  BarChart3,
  BrainCircuit,
  Database,
  Gauge,
  GitCompare,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import AppShell from "@/components/AppShell";
import StatCard from "@/components/StatCard";
import StatusBadge from "@/components/StatusBadge";
import { apiFetch, mediaUrl } from "@/app/lib/api";

type FileInfo = {
  path?: string;
  exists: boolean;
  url?: string | null;
};

type MotorMetrics = {
  task?: string;
  label_0?: string;
  label_1?: string;
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
  confusion_matrix?: number[][];
  num_trials?: number;
  num_left_hand_trials?: number;
  num_right_hand_trials?: number;
  model?: string;
  dataset?: string;
  subject?: number;
  runs?: number[];
  low_freq?: number;
  high_freq?: number;
  tmin?: number;
  tmax?: number;
  num_csp_components?: number;
  num_channels?: number;
  num_samples_per_trial?: number;
  sampling_frequency?: number;
  num_train_trials?: number;
  num_test_trials?: number;
  num_cv_splits?: number;
};

type MotorStatus = {
  module?: string;
  status?: string;
  description?: string;
  dataset_note?: string;
  labels?: Record<string, string>;

  pipeline?: string[];

  metrics?: MotorMetrics | null;

  outputs?: {
    metrics_json?: FileInfo;
    predictions_csv?: FileInfo;
    confusion_matrix?: FileInfo;
    csp_topomap?: FileInfo;
    model?: FileInfo;
    model_file?: FileInfo;
  };

  subject_search?: {
    available?: boolean;
    best_subject?: MotorMetrics | null;
    comparison_csv?: FileInfo;
    comparison_chart?: FileInfo;
  };
};

export default function MotorImageryDashboard() {
  const [status, setStatus] = useState<MotorStatus | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadStatus() {
      setLoading(true);

      const data = await apiFetch<MotorStatus>("/api/motor/status");

      setStatus(data);
      setLoading(false);
    }

    loadStatus();
  }, []);

  const metrics = status?.metrics;
  const bestSubject = status?.subject_search?.best_subject ?? null;

  const confusionMatrixUrl = useMemo(() => {
    const url = status?.outputs?.confusion_matrix?.url;

    if (url) {
      return mediaUrl(url);
    }

    return mediaUrl("motor_imagery/motor_imagery_confusion_matrix.png");
  }, [status]);

  const cspTopomapUrl = useMemo(() => {
    const url = status?.outputs?.csp_topomap?.url;

    if (url) {
      return mediaUrl(url);
    }

    return mediaUrl("motor_imagery/motor_imagery_csp_topomap.png");
  }, [status]);

  const subjectComparisonChartUrl = useMemo(() => {
    const url = status?.subject_search?.comparison_chart?.url;

    if (url) {
      return mediaUrl(url);
    }

    return mediaUrl(
      "motor_imagery/physionet_subject_comparison_accuracy.png",
    );
  }, [status]);

  const modelFile =
    status?.outputs?.model_file ?? status?.outputs?.model;

  return (
    <AppShell
      activePath="/motor"
      title="EEG Motor Imagery BCI"
      subtitle="PhysioNet EEGBCI analysis using CSP + LDA"
    >
      <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-6">
        <StatCard
          icon={Gauge}
          label="Accuracy"
          value={formatMetric(metrics?.accuracy)}
          description="Current selected subject"
          iconClassName="bg-emerald-50 text-emerald-600"
        />

        <StatCard
          icon={BarChart3}
          label="F1 Score"
          value={formatMetric(metrics?.f1_score)}
          description="Classification balance"
          iconClassName="bg-teal-50 text-teal-600"
        />

        <StatCard
          icon={GitCompare}
          label="Best subject"
          value={formatCount(bestSubject?.subject)}
          description="PhysioNet subject search"
          iconClassName="bg-blue-50 text-blue-600"
        />

        <StatCard
          icon={Database}
          label="Channels"
          value={formatCount(metrics?.num_channels)}
          description="EEG channels"
          iconClassName="bg-cyan-50 text-cyan-600"
        />

        <StatCard
          icon={BrainCircuit}
          label="CSP filters"
          value={formatCount(metrics?.num_csp_components)}
          description="Spatial components"
          iconClassName="bg-violet-50 text-violet-600"
        />

        <StatCard
          icon={Activity}
          label="Status"
          value={loading ? "Loading" : status?.status ?? "N/A"}
          description="Backend output check"
          iconClassName="bg-amber-50 text-amber-600"
        />
      </section>

      <section className="mt-5 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h2 className="text-lg font-semibold text-slate-900">
              Motor imagery classification
            </h2>

            <p className="mt-1 text-sm text-slate-500">
              Left-hand versus right-hand EEG motor imagery classification
              using CSP spatial filtering and Linear Discriminant Analysis.
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
              ["Recall", formatMetric(metrics?.recall)],
              ["Specificity", formatMetric(metrics?.specificity)],
              ["Balanced accuracy", formatMetric(metrics?.balanced_accuracy)],
              ["Total trials", formatCount(metrics?.num_trials)],
            ]}
          />
        </div>
      </section>

      <section className="mt-5 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h2 className="text-lg font-semibold text-slate-900">
              CSP Spatial Pattern Topomap
            </h2>

            <p className="mt-1 text-sm text-slate-500">
              CSP topomap shows spatial EEG patterns used to separate
              left-hand and right-hand motor imagery.
            </p>
          </div>

          <StatusBadge active={status?.outputs?.csp_topomap?.exists ?? false} />
        </div>

        <div className="mt-5 grid gap-4 lg:grid-cols-2">
          <ImagePanel
            title="CSP spatial pattern topomap"
            src={cspTopomapUrl}
            compact
          />

          <InfoPanel
            title="Topomap interpretation"
            values={[
              ["Method", "Common Spatial Patterns"],
              ["Purpose", "Separate left/right imagery"],
              ["Input", "Band-passed EEG epochs"],
              ["Feature", "Log-variance of CSP components"],
              ["Classifier", metrics?.model ?? "CSP + LDA"],
              [
                "Topomap file",
                status?.outputs?.csp_topomap?.exists ? "Saved" : "Missing",
              ],
            ]}
          />
        </div>
      </section>

      <section className="mt-5 grid gap-5 lg:grid-cols-2">
        <InfoPanel
          title="Model details"
          values={[
            ["Model", metrics?.model ?? "CSP + LDA"],
            ["Dataset", metrics?.dataset ?? "N/A"],
            ["Subject", formatCount(metrics?.subject)],
            ["Runs", metrics?.runs?.join(", ") ?? "N/A"],
            ["CSP components", formatCount(metrics?.num_csp_components)],
            ["Sampling frequency", formatHz(metrics?.sampling_frequency)],
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
              status?.outputs?.predictions_csv?.exists ? "Saved" : "Missing",
            ],
            ["Model file", modelFile?.exists ? "Saved" : "Missing"],
          ]}
        />
      </section>

      <section className="mt-5 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h2 className="text-lg font-semibold text-slate-900">
              PhysioNet subject search
            </h2>

            <p className="mt-1 text-sm text-slate-500">
              Multiple PhysioNet EEGBCI subjects are compared to select the
              best-performing subject for dashboard visualization.
            </p>
          </div>

          <StatusBadge active={status?.subject_search?.available ?? false} />
        </div>

        <div className="mt-5 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          <MiniMetric
            label="Best subject"
            value={formatCount(bestSubject?.subject)}
          />

          <MiniMetric
            label="Best accuracy"
            value={formatMetric(bestSubject?.accuracy)}
          />

          <MiniMetric
            label="Best F1 score"
            value={formatMetric(bestSubject?.f1_score)}
          />

          <MiniMetric
            label="CV splits"
            value={formatCount(bestSubject?.num_cv_splits)}
          />

          <MiniMetric
            label="Band-pass"
            value={`${formatNumber(bestSubject?.low_freq, 0)}–${formatNumber(
              bestSubject?.high_freq,
              0,
            )} Hz`}
          />

          <MiniMetric
            label="Epoch window"
            value={`${formatNumber(bestSubject?.tmin, 1)}–${formatNumber(
              bestSubject?.tmax,
              1,
            )} s`}
          />

          <MiniMetric
            label="Channels"
            value={formatCount(bestSubject?.num_channels)}
          />

          <MiniMetric
            label="Comparison CSV"
            value={
              status?.subject_search?.comparison_csv?.exists
                ? "Saved"
                : "Missing"
            }
          />
        </div>

        <div className="mt-5">
          <ImagePanel
            title="Subject accuracy comparison"
            src={subjectComparisonChartUrl}
            compact
          />
        </div>
      </section>

      <section className="mt-5 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
        <h2 className="text-lg font-semibold text-slate-900">
          CSP + LDA processing pipeline
        </h2>

        <p className="mt-1 text-sm text-slate-500">
          The current pipeline follows a standard motor imagery BCI workflow.
        </p>

        <div className="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-4 xl:grid-cols-7">
          {(status?.pipeline ?? [
            "EEG trials",
            "Band-pass filtering",
            "Epoch extraction",
            "CSP spatial filtering",
            "Log-variance features",
            "Linear Discriminant Analysis",
            "Left/right imagery prediction",
          ]).map((step, index) => (
            <div
              key={step}
              className="rounded-xl border border-slate-200 bg-white p-4"
            >
              <p className="text-xs font-semibold text-emerald-600">
                {`0${index + 1}`}
              </p>

              <p className="mt-2 text-sm font-semibold text-slate-800">
                {step}
              </p>
            </div>
          ))}
        </div>
      </section>

      <section className="mt-5 rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-800">
        This model is for research and educational use only and is not intended
        for clinical diagnosis.
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
      <h2 className="text-lg font-semibold text-slate-900">{title}</h2>

      <dl className="mt-4 divide-y divide-slate-100">
        {values.map(([label, value]) => (
          <MetricRow key={label} label={label} value={value} />
        ))}
      </dl>
    </article>
  );
}

function MiniMetric({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4">
      <p className="text-xs text-slate-500">{label}</p>

      <p className="mt-1 text-sm font-semibold text-slate-800">
        {value}
      </p>
    </div>
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

      <dd className="text-sm font-semibold text-slate-800">
        {value}
      </dd>
    </div>
  );
}

function formatMetric(value?: number): string {
  return value == null ? "N/A" : value.toFixed(4);
}

function formatNumber(value?: number, digits: number = 2): string {
  return value == null ? "N/A" : value.toFixed(digits);
}

function formatCount(value?: number): string {
  return value == null ? "N/A" : value.toLocaleString();
}

function formatHz(value?: number): string {
  return value == null ? "N/A" : `${value.toFixed(1)} Hz`;
}