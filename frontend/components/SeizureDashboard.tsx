"use client";

import {
  Activity,
  BrainCircuit,
  Database,
  FileHeart,
  Waves,
} from "lucide-react";
import { useMemo, useState } from "react";

import StatCard from "@/components/StatCard";
import StatusBadge from "@/components/StatusBadge";
import { mediaUrl } from "@/lib/api";

type FileInfo = {
  path: string;
  exists: boolean;
};

type SeizureMode = {
  model_file: FileInfo;
  waveform: FileInfo;
  timeline: FileInfo;
  confusion_matrix: FileInfo;

  dataset?: string;
  file?: string;
  subject?: string;
  known_seizure_interval_sec?: number[];
};

export type SeizureStatus = {
  module: string;
  model: string;
  features: string[];

  modes: {
    synthetic: SeizureMode;
    chbmit_one_file: SeizureMode;
    chbmit_multi_file: SeizureMode;
  };

  disclaimer: string;
};

type ModeKey =
  | "synthetic"
  | "chbmit_one_file"
  | "chbmit_multi_file";

type ModeDefinition = {
  key: ModeKey;
  title: string;
  shortTitle: string;
  description: string;
};

const MODE_DEFINITIONS: ModeDefinition[] = [
  {
    key: "synthetic",
    title: "Synthetic Demo EEG",
    shortTitle: "Synthetic",
    description:
      "Synthetic EEG-like signals used to verify the feature extraction and classification pipeline.",
  },
  {
    key: "chbmit_one_file",
    title: "CHB-MIT One-File EEG",
    shortTitle: "One file",
    description:
      "A real EEG prototype based on one seizure-containing CHB-MIT recording.",
  },
  {
    key: "chbmit_multi_file",
    title: "CHB-MIT Multi-File EEG",
    shortTitle: "Multi-file",
    description:
      "A subject-level prototype trained and evaluated using multiple CHB-MIT EEG recordings.",
  },
];

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
    confusionMatrix:
      "seizure/chbmit_chb01_03_confusion_matrix.png",
  },

  chbmit_multi_file: {
    waveform: "seizure/chbmit_multi_file_waveform.png",
    timeline:
      "seizure/chbmit_multi_file_probability_timeline.png",
    confusionMatrix:
      "seizure/chbmit_multi_file_confusion_matrix.png",
  },
};

export default function SeizureDashboard({
  status,
}: {
  status: SeizureStatus;
}) {
  const [selectedMode, setSelectedMode] =
    useState<ModeKey>("chbmit_multi_file");

  const mode = status.modes[selectedMode];
  const filenames = RESULT_FILENAMES[selectedMode];

  const definition = useMemo(
    () =>
      MODE_DEFINITIONS.find(
        (item) => item.key === selectedMode,
      ) ?? MODE_DEFINITIONS[2],
    [selectedMode],
  );

  const datasetName =
    mode.dataset ??
    (selectedMode === "synthetic"
      ? "Synthetic EEG"
      : "CHB-MIT");

  const sourceDescription = getSourceDescription(
    selectedMode,
    mode,
  );

  return (
    <>
      <ModeSelector
        selectedMode={selectedMode}
        onSelect={setSelectedMode}
      />

      <section className="mt-5 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard
          icon={BrainCircuit}
          label="Model"
          value={status.model}
          description={
            mode.model_file.exists
              ? "Model available"
              : "Model missing"
          }
          iconClassName="bg-blue-50 text-blue-600"
        />

        <StatCard
          icon={Database}
          label="Dataset"
          value={datasetName}
          description={definition.shortTitle}
          iconClassName="bg-emerald-50 text-emerald-600"
        />

        <StatCard
          icon={Activity}
          label="Analysis mode"
          value={definition.shortTitle}
          description="Selected EEG source"
          iconClassName="bg-violet-50 text-violet-600"
        />

        <StatCard
          icon={Waves}
          label="Features"
          value={String(status.features.length)}
          description="EEG feature groups"
          iconClassName="bg-teal-50 text-teal-600"
        />
      </section>

      <section className="mt-5 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
        <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-start">
          <div>
            <p className="text-xs font-semibold uppercase tracking-wide text-teal-600">
              Selected source
            </p>

            <h2 className="mt-1 text-xl font-semibold text-slate-900">
              {definition.title}
            </h2>

            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-500">
              {definition.description}
            </p>
          </div>

          <StatusBadge
            active={mode.model_file.exists}
            label={
              mode.model_file.exists
                ? "Model available"
                : "Model missing"
            }
          />
        </div>

        <div className="mt-5 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          <SourceValue
            label="Dataset"
            value={datasetName}
          />

          <SourceValue
            label="Recording"
            value={
              mode.file ??
              mode.subject ??
              (selectedMode === "synthetic"
                ? "Generated signal"
                : "N/A")
            }
          />

          <SourceValue
            label="Mode"
            value={definition.shortTitle}
          />

          <SourceValue
            label="Known seizure interval"
            value={
              mode.known_seizure_interval_sec
                ? `${mode.known_seizure_interval_sec[0]}–${mode.known_seizure_interval_sec[1]} sec`
                : "Not fixed"
            }
          />
        </div>
      </section>

      <section className="mt-5 grid gap-5 xl:grid-cols-[1.35fr_0.65fr]">
        <article className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div>
            <h2 className="text-lg font-semibold text-slate-900">
              EEG signal analysis
            </h2>

            <p className="mt-1 text-sm text-slate-500">
              Waveform and seizure-probability results for the
              selected data source.
            </p>
          </div>

          <div className="mt-5 space-y-4">
            <ResultImage
              title="EEG waveform"
              description="Input EEG signal used by the selected analysis mode."
              src={mediaUrl(filenames.waveform)}
              available={mode.waveform.exists}
            />

            <ResultImage
              title="Seizure probability timeline"
              description="Window-level seizure probability produced by the classifier."
              src={mediaUrl(filenames.timeline)}
              available={mode.timeline.exists}
            />
          </div>
        </article>

        <aside className="space-y-5">
          <article className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
            <div className="flex items-center justify-between gap-4">
              <div>
                <h2 className="text-lg font-semibold text-slate-900">
                  Classification result
                </h2>

                <p className="mt-1 text-sm text-slate-500">
                  Confusion matrix for the selected model.
                </p>
              </div>

              <StatusBadge
                active={mode.confusion_matrix.exists}
                label={
                  mode.confusion_matrix.exists
                    ? "Available"
                    : "Missing"
                }
              />
            </div>

            <div className="mt-4">
              <ResultImage
                title="Confusion matrix"
                description="Predicted and true seizure classes."
                src={mediaUrl(
                  filenames.confusionMatrix,
                )}
                available={
                  mode.confusion_matrix.exists
                }
                compact
              />
            </div>
          </article>

          <article className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-blue-50 text-blue-600">
                <FileHeart size={21} />
              </div>

              <div>
                <h2 className="text-lg font-semibold text-slate-900">
                  EEG features
                </h2>

                <p className="text-sm text-slate-500">
                  Features used by the classifier
                </p>
              </div>
            </div>

            <div className="mt-4 flex flex-wrap gap-2">
              {status.features.map((feature) => (
                <span
                  key={feature}
                  className="rounded-full bg-blue-50 px-3 py-1.5 text-xs font-medium text-blue-700"
                >
                  {feature}
                </span>
              ))}
            </div>
          </article>
        </aside>
      </section>

      <section className="mt-5 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
        <h2 className="text-lg font-semibold text-slate-900">
          Analysis pipeline
        </h2>

        <div className="mt-5 grid gap-3 sm:grid-cols-2 xl:grid-cols-5">
          <PipelineStep number="1" label="EEG input" />
          <PipelineStep number="2" label="Sliding windows" />
          <PipelineStep number="3" label="Feature extraction" />
          <PipelineStep number="4" label="Random Forest" />
          <PipelineStep
            number="5"
            label="Probability timeline"
          />
        </div>

        <p className="mt-5 text-sm leading-6 text-slate-500">
          {sourceDescription}
        </p>
      </section>

      <section className="mt-5 rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm leading-6 text-amber-800">
        {status.disclaimer}
      </section>
    </>
  );
}

function ModeSelector({
  selectedMode,
  onSelect,
}: {
  selectedMode: ModeKey;
  onSelect: (mode: ModeKey) => void;
}) {
  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex flex-col justify-between gap-3 sm:flex-row sm:items-center">
        <div>
          <h2 className="text-base font-semibold text-slate-900">
            Select EEG data source
          </h2>

          <p className="mt-1 text-sm text-slate-500">
            Compare synthetic and real CHB-MIT seizure-analysis
            modes.
          </p>
        </div>

        <span className="text-xs font-medium text-slate-400">
          3 modes available
        </span>
      </div>

      <div
        className="mt-4 grid gap-2 md:grid-cols-3"
        role="tablist"
        aria-label="EEG data source"
      >
        {MODE_DEFINITIONS.map((definition) => {
          const selected =
            definition.key === selectedMode;

          return (
            <button
              key={definition.key}
              type="button"
              role="tab"
              aria-selected={selected}
              onClick={() =>
                onSelect(definition.key)
              }
              className={[
                "rounded-xl border px-4 py-3 text-left transition",
                "focus:outline-none focus:ring-2 focus:ring-teal-500 focus:ring-offset-2",
                selected
                  ? "border-teal-600 bg-teal-600 text-white shadow-sm"
                  : "border-slate-200 bg-white text-slate-700 hover:border-teal-300 hover:bg-teal-50",
              ].join(" ")}
            >
              <p className="text-sm font-semibold">
                {definition.title}
              </p>

              <p
                className={[
                  "mt-1 text-xs",
                  selected
                    ? "text-teal-50"
                    : "text-slate-500",
                ].join(" ")}
              >
                {getShortModeDescription(
                  definition.key,
                )}
              </p>
            </button>
          );
        })}
      </div>
    </section>
  );
}

function ResultImage({
  title,
  description,
  src,
  available,
  compact = false,
}: {
  title: string;
  description: string;
  src: string;
  available: boolean;
  compact?: boolean;
}) {
  return (
    <figure className="overflow-hidden rounded-xl border border-slate-200">
      <figcaption className="border-b border-slate-200 bg-slate-50 px-4 py-3">
        <p className="text-sm font-medium text-slate-700">
          {title}
        </p>

        <p className="mt-0.5 text-xs text-slate-500">
          {description}
        </p>
      </figcaption>

      <div
        className={[
          "flex items-center justify-center bg-slate-950 p-3",
          compact ? "min-h-64" : "min-h-72",
        ].join(" ")}
      >
        {available ? (
          <img
            src={src}
            alt={title}
            className={[
              "w-full object-contain",
              compact ? "max-h-72" : "max-h-96",
            ].join(" ")}
          />
        ) : (
          <div className="px-6 py-12 text-center">
            <p className="text-sm font-medium text-slate-300">
              Result image not found
            </p>

            <p className="mt-2 text-xs text-slate-500">
              Run the corresponding training and inference
              scripts first.
            </p>
          </div>
        )}
      </div>
    </figure>
  );
}

function SourceValue({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="rounded-xl border border-slate-200 bg-slate-50 p-3">
      <p className="text-xs font-medium text-slate-500">
        {label}
      </p>

      <p
        className="mt-1 truncate text-sm font-semibold text-slate-800"
        title={value}
      >
        {value}
      </p>
    </div>
  );
}

function PipelineStep({
  number,
  label,
}: {
  number: string;
  label: string;
}) {
  return (
    <div className="flex items-center gap-3 rounded-xl bg-slate-50 p-4">
      <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-teal-600 text-xs font-bold text-white">
        {number}
      </span>

      <span className="text-sm font-medium text-slate-700">
        {label}
      </span>
    </div>
  );
}

function getShortModeDescription(
  mode: ModeKey,
): string {
  switch (mode) {
    case "synthetic":
      return "Pipeline validation";

    case "chbmit_one_file":
      return "Single real EDF";

    case "chbmit_multi_file":
      return "Multiple real EDFs";
  }
}

function getSourceDescription(
  mode: ModeKey,
  data: SeizureMode,
): string {
  switch (mode) {
    case "synthetic":
      return (
        "Synthetic mode is intended to verify the complete EEG " +
        "feature-extraction, classification, and visualization pipeline."
      );

    case "chbmit_one_file":
      return (
        `One-file mode analyzes ${data.file ?? "one CHB-MIT EDF recording"}. ` +
        "It is useful for confirming that real EDF loading and seizure-interval labeling work correctly."
      );

    case "chbmit_multi_file":
      return (
        `Multi-file mode uses subject ${data.subject ?? "chb01"} across multiple CHB-MIT recordings. ` +
        "It is a subject-level research prototype rather than a patient-independent clinical evaluation."
      );
  }
}