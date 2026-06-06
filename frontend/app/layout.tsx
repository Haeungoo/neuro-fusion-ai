import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "NeuroFusion-AI",
  description:
    "Multimodal neuroscience AI platform for MRI and EEG analysis.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}