export type TextureSource =
  | ImageBitmap
  | HTMLImageElement
  | HTMLCanvasElement
  | OffscreenCanvas;

export interface ModelMaterial {
  color: [number, number, number];
  map: TextureSource | null;
  wrapS: GPUAddressMode;
  wrapT: GPUAddressMode;
  tintWithObjectColor: boolean;
}

export function createDefaultMaterial(): ModelMaterial {
  return {
    color: [1, 1, 1],
    map: null,
    wrapS: "repeat",
    wrapT: "repeat",
    tintWithObjectColor: true,
  };
}
