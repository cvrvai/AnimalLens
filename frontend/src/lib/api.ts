import { Dataset, DatasetUploadResponse, Keyframe } from '@/types/dataset';
import { SpeciesItem } from '@/types/species';
import { StartTrainingRequest, TrainingJob } from '@/types/training';

const BASE_URL = process.env.NEXT_PUBLIC_API_URL || '/api/backend';

export async function fetchHealth() {
  const res = await fetch(`${BASE_URL}/health`);
  if (!res.ok) throw new Error('Failed to connect to backend server');
  return res.json();
}

export async function fetchSpecies(): Promise<SpeciesItem[]> {
  const res = await fetch(`${BASE_URL}/species`);
  if (!res.ok) throw new Error('Failed to fetch species list');
  return res.json();
}

export async function fetchDatasets(): Promise<Dataset[]> {
  const res = await fetch(`${BASE_URL}/datasets`);
  if (!res.ok) throw new Error('Failed to fetch datasets');
  return res.json();
}

export async function uploadDatasetVideo(formData: FormData): Promise<DatasetUploadResponse> {
  const res = await fetch(`${BASE_URL}/datasets/upload-video`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed to upload video' }));
    throw new Error(err.detail || 'Upload failed');
  }
  return res.json();
}

export async function fetchDatasetFrames(datasetName: string): Promise<{ dataset_name: string; total_frames: number; frames: Keyframe[] }> {
  const res = await fetch(`${BASE_URL}/datasets/${datasetName}/frames`);
  if (!res.ok) throw new Error(`Failed to fetch frames for ${datasetName}`);
  return res.json();
}

export async function updateFrameAnnotation(datasetName: string, imageName: string, split: string, bboxes: any[]) {
  const res = await fetch(`${BASE_URL}/datasets/${datasetName}/annotations`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image_name: imageName, split, bboxes }),
  });
  if (!res.ok) throw new Error('Failed to save annotations');
  return res.json();
}

export async function fetchTrainingJobs(): Promise<TrainingJob[]> {
  const res = await fetch(`${BASE_URL}/train/jobs`);
  if (!res.ok) throw new Error('Failed to fetch training jobs');
  return res.json();
}

export async function startTrainingJob(payload: StartTrainingRequest): Promise<TrainingJob> {
  const res = await fetch(`${BASE_URL}/train/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed to start training' }));
    throw new Error(err.detail || 'Training failed to start');
  }
  return res.json();
}

export async function fetchTrainingJobStatus(jobId: string): Promise<TrainingJob> {
  const res = await fetch(`${BASE_URL}/train/status/${jobId}`);
  if (!res.ok) throw new Error(`Failed to fetch status for job ${jobId}`);
  return res.json();
}

export async function fetchModels() {
  const res = await fetch(`${BASE_URL}/models`);
  if (!res.ok) throw new Error('Failed to fetch models');
  return res.json();
}
