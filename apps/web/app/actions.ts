"use server";

import { revalidatePath } from "next/cache";

import { submitRun } from "@/lib/api";

export async function submitRunAction(
  objective: string,
): Promise<{ ok: true; requestId: string } | { ok: false; error: string }> {
  try {
    const result = await submitRun(objective);
    revalidatePath("/");
    return { ok: true, requestId: result.request_id };
  } catch (error) {
    return { ok: false, error: error instanceof Error ? error.message : "Submission failed." };
  }
}
