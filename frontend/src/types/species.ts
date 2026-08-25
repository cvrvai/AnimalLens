export interface SpeciesConfig {
  id: string;
  name: string;
  scientific_name: string;
  taxonomy_version: string;
  default_model: string;
  detection_threshold: number;
  classes: string[];
}

export interface BehaviorCategory {
  name: string;
  description?: string;
  labels: string[];
}

export interface BehaviorTaxonomy {
  version: string;
  species: string;
  categories: Record<string, BehaviorCategory>;
}

export interface SpeciesItem {
  id: string;
  name: string;
  scientific_name: string;
  taxonomy_version: string;
  default_model: string;
  behavior_categories: string[];
}
