import { BrainCircuit, Database, Gauge, ScanLine } from "lucide-react";

import AppShell from "@/components/AppShell";
import StatCard from "@/components/StatCard";
import StatusBadge from "@/components/StatusBadge";
import { apiFetch, mediaUrl } from "@/lib/api";

type FileInfo = {
  path: string;
  exists: boolean;
};

type MriStatus = {
  module: string;
  dataset: string;
  model: string;
  model_file: FileInfo;

  outputs: {
    input_slice: FileInfo;
    ground_truth_mask: FileInfo;
    predicted_mask: FileInfo;
    overlay: FileInfo;
  };

  training_metrics: {
    best_val_loss?: number;
    epochs?: number;
    num_samples?: number;
    train_samples?: number;
    val_samples?: number;
  };

  inference_metrics: {
    sample_name?: string;
    dice_score?: number;
    threshold?: number;
    prediction_pixels?: number;
    ground_truth_pixels?: number;
  };

  validation_metrics: {
    evaluation_type?: string;
    threshold?: number;
    num_validation_cases?: number;
    num_validation_slices?: number;
    validation_cases?: string[];

    slice_level?: {
      mean_dice?: number;
      mean_iou?: number;
      mean_precision?: number;
      mean_recall?: number;
      mean_specificity?: number;
      mean_accuracy?: number;
    };

    case_level?: {
      mean_dice?: number;
      mean_iou?: number;
      mean_precision?: number;
      mean_recall?: number;
      mean_specificity?: number;
      mean_accuracy?: number;
    };
  };

  disclaimer: string;
};

export default async function MriPage() {
  const status = await apiFetch<MriStatus>("/api/mri/status");

  const caseMeanDice =
    status?.validation_metrics?.case_level?.mean_dice;

  const caseMeanIoU =
    status?.validation_metrics?.case_level?.mean_iou;

  const sliceMeanDice =
    status?.validation_metrics?.slice_level?.mean_dice;

  const sliceMeanIoU =
    status?.validation_metrics?.slice_level?.mean_iou;

  const validationCases =
    status?.validation_metrics?.num_validation_cases;

  const validationSlices =
    status?.validation_metrics?.num_validation_slices;

  return (
    <AppShell
      activePath="/mri"
      title="MRI Tumor Segmentation"
      subtitle="BraTS MRI analysis using a 2D U-Net"
    >
      <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-6">
        <StatCard
          icon={Gauge}
          label="Case mean Dice"
          value={formatMetric(caseMeanDice)}
          description="Patient-level validation"
          iconClassName="bg-violet-50 text-violet-600"
        />
        <StatCard
          icon={ScanLine}
          label="Case mean IoU"
          value={formatMetric(caseMeanIoU)}
          description="Patient-level validation"
          iconClassName="bg-blue-50 text-blue-600"
        />
        <StatCard
          icon={Gauge}
          label="Slice mean Dice"
          value={formatMetric(sliceMeanDice)}
          description="All validation slices"
          iconClassName="bg-teal-50 text-teal-600"
        />
        <StatCard
          icon={ScanLine}
          label="Slice mean IoU"
          value={formatMetric(sliceMeanIoU)}
          description="All validation slices"
          iconClassName="bg-cyan-50 text-cyan-600"
        />
        <StatCard
          icon={Database}
          label="Validation cases"
          value={formatCount(validationCases)}
          description="Unseen BraTS cases"
          iconClassName="bg-emerald-50 text-emerald-600"
        />
        <StatCard
          icon={BrainCircuit}
          label="Validation slices"
          value={formatCount(validationSlices)}
          description="Evaluated MRI slices"
          iconClassName="bg-amber-50 text-amber-600"
        />
      </section>

      <section className="mt-5 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h2 className="text-lg font-semibold text-slate-900">
              Segmentation comparison
            </h2>
            <p className="mt-1 text-sm text-slate-500">
              Original MRI, annotation, model prediction, and final overlay.
            </p>
          </div>
          <StatusBadge active={status?.model_file.exists ?? false} />
        </div>

        <div className="mt-5 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          <ImagePanel
            title="Input MRI"
            src={mediaUrl("mri/mri_input_slice.png")}
          />
          <ImagePanel
            title="Ground truth"
            src={mediaUrl("mri/mri_ground_truth_mask.png")}
          />
          <ImagePanel
            title="Predicted mask"
            src={mediaUrl("mri/mri_predicted_mask.png")}
          />
          <ImagePanel
            title="Prediction overlay"
            src={mediaUrl("mri/mri_prediction_overlay.png")}
          />
        </div>
      </section>

      <section className="mt-5 grid gap-5 lg:grid-cols-2">
        <InfoPanel
          title="Inference details"
          values={[
            ["Sample", status?.inference_metrics?.sample_name ?? "N/A"],
            [
              "Threshold",
              String(status?.inference_metrics?.threshold ?? "N/A"),
            ],
            [
              "Predicted pixels",
              formatMetric(status?.inference_metrics?.prediction_pixels),
            ],
            [
              "Ground-truth pixels",
              formatMetric(status?.inference_metrics?.ground_truth_pixels),
            ],
          ]}
        />

        <section className="mt-5 grid gap-5 lg:grid-cols-2">
            <article className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                <h2 className="text-lg font-semibold text-slate-900">
                Patient-level validation
                </h2>

                <p className="mt-1 text-sm text-slate-500">
                Metrics averaged across unseen BraTS cases.
                </p>

                <dl className="mt-4 divide-y divide-slate-100">
                <MetricRow
                    label="Mean Dice"
                    value={formatMetric(caseMeanDice)}
                />
                <MetricRow
                    label="Mean IoU"
                    value={formatMetric(caseMeanIoU)}
                />
                <MetricRow
                    label="Mean precision"
                    value={formatMetric(
                    status?.validation_metrics?.case_level?.mean_precision,
                    )}
                />
                <MetricRow
                    label="Mean recall"
                    value={formatMetric(
                    status?.validation_metrics?.case_level?.mean_recall,
                    )}
                />
                <MetricRow
                    label="Validation cases"
                    value={formatCount(validationCases)}
                />
                </dl>
            </article>

            <article className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                <h2 className="text-lg font-semibold text-slate-900">
                Slice-level validation
                </h2>

                <p className="mt-1 text-sm text-slate-500">
                Metrics averaged over every validation MRI slice.
                </p>

                <dl className="mt-4 divide-y divide-slate-100">
                <MetricRow
                    label="Mean Dice"
                    value={formatMetric(sliceMeanDice)}
                />
                <MetricRow
                    label="Mean IoU"
                    value={formatMetric(sliceMeanIoU)}
                />
                <MetricRow
                    label="Mean precision"
                    value={formatMetric(
                    status?.validation_metrics?.slice_level?.mean_precision,
                    )}
                />
                <MetricRow
                    label="Mean recall"
                    value={formatMetric(
                    status?.validation_metrics?.slice_level?.mean_recall,
                    )}
                />
                <MetricRow
                    label="Validation slices"
                    value={formatCount(validationSlices)}
                />
                </dl>
            </article>
        </section>

        <InfoPanel
          title="Training details"
          values={[
            [
              "Epochs",
              String(status?.training_metrics?.epochs ?? "N/A"),
            ],
            [
              "Total slices",
              formatMetric(status?.training_metrics?.num_samples),
            ],
            [
              "Training slices",
              formatMetric(status?.training_metrics?.train_samples),
            ],
            [
              "Validation slices",
              formatMetric(status?.training_metrics?.val_samples),
            ],
          ]}
        />
      </section>

      <section className="mt-5 rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-800">
        {status?.disclaimer ??
          "This model is for research and educational use only."}
      </section>
    </AppShell>
  );
}

function ImagePanel({ title, src }: { title: string; src: string }) {
  return (
    <figure className="overflow-hidden rounded-xl border border-slate-200">
      <div className="flex h-56 items-center justify-center bg-slate-950 p-3">
        <img src={src} alt={title} className="max-h-full max-w-full object-contain" />
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
          <div
            key={label}
            className="flex items-center justify-between gap-5 py-3"
          >
            <dt className="text-sm text-slate-500">{label}</dt>
            <dd className="truncate text-sm font-semibold text-slate-800">
              {value}
            </dd>
          </div>
        ))}
      </dl>
    </article>
  );
}

function formatMetric(value?: number): string {
  return value == null ? "N/A" : value.toFixed(4);
}

function formatCount(value?: number): string {
  return value == null ? "N/A" : value.toLocaleString();
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