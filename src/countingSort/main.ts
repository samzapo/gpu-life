import type { UnifiedEngineRefs } from '../engineTypes';
import {
  linkComputeTimestamp,
  readTimestamp,
  resolveTimestamp,
  setupTimestamp,
} from '../utils';

import structsShader from '../shaders/structs.wgsl?raw';
import projectionShader from '../shaders/projection.wgsl?raw';
import bioChemShader from '../shaders/bioChem.wgsl?raw';
import valenceSimShader from '../shaders/valenceSim.wgsl?raw';
import oceanSimHelperShader from '../shaders/oceanSim.wgsl?raw';
import cellShader from './cell.wgsl?raw';
import prefixShader from './prefix.wgsl?raw';
import sortShader from './sort.wgsl?raw';
import valenceShader from './valence.wgsl?raw';
import bioStepShader from './bioStep.wgsl?raw';
import simShader from './sim.wgsl?raw';

interface CountingWorldState {
  cellBuffer: GPUBuffer;
  countBuffer: GPUBuffer;
  zeroBuffer: GPUBuffer;
  sortedBuffer: GPUBuffer;
  indicesBuffer: GPUBuffer;
  cellBindGroups: [GPUBindGroup, GPUBindGroup];
  prefixBindGroup: GPUBindGroup;
  sortBindGroup: GPUBindGroup;
  valenceBindGroups: [GPUBindGroup, GPUBindGroup];
  bioBindGroups: [GPUBindGroup, GPUBindGroup];
  simBindGroups: [GPUBindGroup, GPUBindGroup];
}

const workgroupSize = 128;

let cellPipeline: GPUComputePipeline | undefined;
let prefixPipeline: GPUComputePipeline | undefined;
let sortPipeline: GPUComputePipeline | undefined;
let valencePipeline: GPUComputePipeline | undefined;
let bioPipeline: GPUComputePipeline | undefined;
let simPipeline: GPUComputePipeline | undefined;

let worldState: CountingWorldState | undefined;

function destroyState() {
  if (!worldState) return;
  worldState.cellBuffer.destroy();
  worldState.sortedBuffer.destroy();
  worldState.countBuffer.destroy();
  worldState.indicesBuffer.destroy();
  worldState.zeroBuffer.destroy();
  worldState = undefined;
}

export function setup(device: GPUDevice) {
  setupTimestamp(device, 'cell');
  cellPipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ code: `${structsShader}\n${cellShader}` }),
      entryPoint: 'main',
    },
  });

  setupTimestamp(device, 'prefix');
  prefixPipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ code: prefixShader }),
      entryPoint: 'main',
    },
  });

  setupTimestamp(device, 'sort');
  sortPipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ code: `${structsShader}\n${sortShader}` }),
      entryPoint: 'main',
    },
  });

  setupTimestamp(device, 'csValence');
  valencePipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: device.createShaderModule({
        code: `${structsShader}\n${projectionShader}\n${bioChemShader}\n${valenceSimShader}\n${valenceShader}`,
      }),
      entryPoint: 'main',
    },
  });

  bioPipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: device.createShaderModule({
        code: `${structsShader}\n${projectionShader}\n${bioChemShader}\n${bioStepShader}`,
      }),
      entryPoint: 'main',
    },
  });

  setupTimestamp(device, 'countSim');
  simPipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: device.createShaderModule({
        code: `${structsShader}\n${projectionShader}\n${bioChemShader}\n${valenceSimShader}\n${oceanSimHelperShader}\n${simShader}`,
      }),
      entryPoint: 'main',
    },
  });
}

export function start(device: GPUDevice, refs: UnifiedEngineRefs) {
  if (
    !cellPipeline ||
    !prefixPipeline ||
    !sortPipeline ||
    !valencePipeline ||
    !bioPipeline ||
    !simPipeline
  ) {
    return;
  }

  destroyState();

  const { particleAmt, cellAmt } = refs;
  const cellBuffer = device.createBuffer({
    size: particleAmt * 12 * Float32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    label: 'csCellBuffer',
  });
  const sortedBuffer = device.createBuffer({
    size: particleAmt * 12 * Float32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    label: 'csSortedBuffer',
  });
  const countBuffer = device.createBuffer({
    size: cellAmt * Uint32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    label: 'csCountBuffer',
  });
  const indicesBuffer = device.createBuffer({
    size: cellAmt * 2 * Uint32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    label: 'csIndicesBuffer',
  });
  const zeroBuffer = device.createBuffer({
    size: cellAmt * Uint32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    label: 'csZeroBuffer',
  });
  device.queue.writeBuffer(zeroBuffer, 0, new Uint32Array(cellAmt));

  const prefixBindGroup = device.createBindGroup({
    layout: prefixPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: countBuffer } },
      { binding: 1, resource: { buffer: indicesBuffer } },
    ],
  });

  const sortBindGroup = device.createBindGroup({
    layout: sortPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: cellBuffer } },
      { binding: 1, resource: { buffer: sortedBuffer } },
      { binding: 2, resource: { buffer: countBuffer } },
    ],
  });

  const cellGroups: GPUBindGroup[] = [];
  const valenceGroups: GPUBindGroup[] = [];
  const bioGroups: GPUBindGroup[] = [];
  const simGroups: GPUBindGroup[] = [];

  for (let i = 0; i < 2; i++) {
    cellGroups.push(
      device.createBindGroup({
        layout: cellPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: refs.simBuffer } },
          { binding: 1, resource: { buffer: refs.particleBuffers[i] } },
          { binding: 2, resource: { buffer: refs.lifeStateBuffers[i] } },
          { binding: 3, resource: { buffer: cellBuffer } },
          { binding: 4, resource: { buffer: countBuffer } },
        ],
      }),
    );

    valenceGroups.push(
      device.createBindGroup({
        layout: valencePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 1, resource: { buffer: refs.simBuffer } },
          { binding: 2, resource: { buffer: refs.rulesMatrixBuffer } },
          { binding: 3, resource: { buffer: refs.particleBuffers[i] } },
          { binding: 4, resource: { buffer: refs.lifeStateBuffers[i] } },
          { binding: 5, resource: { buffer: refs.valenceBuffer } },
          { binding: 6, resource: { buffer: sortedBuffer } },
          { binding: 7, resource: { buffer: indicesBuffer } },
        ],
      }),
    );

    bioGroups.push(
      device.createBindGroup({
        layout: bioPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: refs.bioUniformBuffer } },
          { binding: 1, resource: refs.oceanTexView },
          { binding: 2, resource: { buffer: refs.rulesMatrixBuffer } },
          { binding: 3, resource: { buffer: refs.particleBuffers[i] } },
          { binding: 4, resource: { buffer: refs.lifeStateBuffers[i] } },
          { binding: 5, resource: { buffer: refs.lifeStateBuffers[1 - i] } },
          { binding: 6, resource: { buffer: refs.valenceBuffer } },
          { binding: 7, resource: { buffer: refs.bioForceBuffer } },
          { binding: 8, resource: { buffer: sortedBuffer } },
          { binding: 9, resource: { buffer: indicesBuffer } },
          { binding: 10, resource: refs.windStackTexView },
          { binding: 11, resource: refs.tempStackTexView },
          { binding: 12, resource: refs.uvTexView },
          { binding: 13, resource: refs.hydroTexView },
          { binding: 14, resource: refs.terrainTexView },
        ],
      }),
    );

    simGroups.push(
      device.createBindGroup({
        layout: simPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: refs.uniformBuffer } },
          { binding: 1, resource: { buffer: refs.simBuffer } },
          { binding: 2, resource: { buffer: refs.oceanSimBuffer } },
          { binding: 3, resource: refs.windStackTexView },
          { binding: 4, resource: refs.oceanTexView },
          { binding: 5, resource: { buffer: refs.rulesMatrixBuffer } },
          { binding: 6, resource: { buffer: refs.particleBuffers[i] } },
          { binding: 7, resource: { buffer: refs.particleBuffers[1 - i] } },
          { binding: 8, resource: { buffer: refs.lifeStateBuffers[1 - i] } },
          { binding: 9, resource: { buffer: refs.valenceBuffer } },
          { binding: 10, resource: { buffer: refs.bioForceBuffer } },
          { binding: 11, resource: { buffer: sortedBuffer } },
          { binding: 12, resource: { buffer: indicesBuffer } },
          { binding: 13, resource: refs.tempStackTexView },
          { binding: 14, resource: refs.uvTexView },
          { binding: 15, resource: refs.hydroTexView },
          { binding: 16, resource: refs.terrainTexView },
        ],
      }),
    );
  }

  worldState = {
    cellBuffer,
    countBuffer,
    zeroBuffer,
    sortedBuffer,
    indicesBuffer,
    cellBindGroups: [cellGroups[0], cellGroups[1]],
    prefixBindGroup,
    sortBindGroup,
    valenceBindGroups: [valenceGroups[0], valenceGroups[1]],
    bioBindGroups: [bioGroups[0], bioGroups[1]],
    simBindGroups: [simGroups[0], simGroups[1]],
  };
}

export function tick(
  device: GPUDevice,
  commandEncoder: GPUCommandEncoder,
  alternate: number,
  particleAmt: number,
) {
  const s = worldState;
  if (
    !s ||
    !cellPipeline ||
    !prefixPipeline ||
    !sortPipeline ||
    !valencePipeline ||
    !bioPipeline ||
    !simPipeline
  ) {
    return;
  }

  const wgCount = Math.ceil(particleAmt / workgroupSize);

  commandEncoder.copyBufferToBuffer(s.zeroBuffer, s.countBuffer);

  const cellPass = commandEncoder.beginComputePass(linkComputeTimestamp(device, 'cell'));
  cellPass.setPipeline(cellPipeline);
  cellPass.setBindGroup(0, s.cellBindGroups[alternate]);
  cellPass.dispatchWorkgroups(wgCount);
  cellPass.end();
  resolveTimestamp(commandEncoder, 'cell');

  const prefixPass = commandEncoder.beginComputePass(
    linkComputeTimestamp(device, 'prefix'),
  );
  prefixPass.setPipeline(prefixPipeline);
  prefixPass.setBindGroup(0, s.prefixBindGroup);
  prefixPass.dispatchWorkgroups(1);
  prefixPass.end();
  resolveTimestamp(commandEncoder, 'prefix');

  const sortPass = commandEncoder.beginComputePass(linkComputeTimestamp(device, 'sort'));
  sortPass.setPipeline(sortPipeline);
  sortPass.setBindGroup(0, s.sortBindGroup);
  sortPass.dispatchWorkgroups(wgCount);
  sortPass.end();
  resolveTimestamp(commandEncoder, 'sort');

  const valencePass = commandEncoder.beginComputePass(
    linkComputeTimestamp(device, 'csValence'),
  );
  valencePass.setPipeline(valencePipeline);
  valencePass.setBindGroup(0, s.valenceBindGroups[alternate]);
  valencePass.dispatchWorkgroups(wgCount);
  valencePass.end();
  resolveTimestamp(commandEncoder, 'csValence');

  const bioPass = commandEncoder.beginComputePass({});
  bioPass.setPipeline(bioPipeline);
  bioPass.setBindGroup(0, s.bioBindGroups[alternate]);
  bioPass.dispatchWorkgroups(wgCount);
  bioPass.end();

  const simPass = commandEncoder.beginComputePass(
    linkComputeTimestamp(device, 'countSim'),
  );
  simPass.setPipeline(simPipeline);
  simPass.setBindGroup(0, s.simBindGroups[alternate]);
  simPass.dispatchWorkgroups(wgCount);
  simPass.end();
  resolveTimestamp(commandEncoder, 'countSim');
}

export function updateDisplays(params: Record<string, number>) {
  readTimestamp('cell').then((time) => {
    params.cell = time;
  });
  readTimestamp('prefix').then((time) => {
    params.prefix = time;
  });
  readTimestamp('sort').then((time) => {
    params.sort = time;
  });
  readTimestamp('csValence').then((time) => {
    params.valence = time;
  });
  readTimestamp('countSim').then((time) => {
    params.sim = time;
  });
}
