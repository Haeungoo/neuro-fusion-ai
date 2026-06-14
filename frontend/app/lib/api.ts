const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ??
  "http://127.0.0.1:8000";

export function mediaUrl(relativePath: string): string {
  const cleanPath = relativePath
    .replace(/^http:\/\/127\.0\.0\.1:8000\/media\/results\//, "")
    .replace(/^http:\/\/localhost:8000\/media\/results\//, "")
    .replace(/^\/media\/results\//, "")
    .replace(/^results\//, "")
    .replace(/^neuro-fusion-ai\//, "");

  return `${API_BASE_URL}/media/results/${cleanPath}`;
}

export async function apiFetch<T>(
  endpoint: string,
): Promise<T | null> {
  try {
    const response = await fetch(
      `${API_BASE_URL}${endpoint}`,
      {
        cache: "no-store",
      },
    );

    if (!response.ok) {
      return null;
    }

    return (await response.json()) as T;
  } catch {
    return null;
  }
}