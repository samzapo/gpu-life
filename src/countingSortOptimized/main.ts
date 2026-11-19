import type { UnifiedEngineRefs } from '../engineTypes';
import {
  linkComputeTimestamp,
  readTimestamp,
  resolveTimestamp,
  setupTimestamp,
} from '../utils';

import { cellComputeShader } from './computeShaders/cell';
import { sortComputeShader } from './computeShaders/sort';
import { valenceComputeShader } from './computeShaders/valence';
import { bioStepComputeShader } from './computeShaders/bio';
import { simComputeShader } from './computeShaders/sim';
import { InclusivePrefixSum } from './InclusivePrefixSum';

interface PrefixWorldState {
  cellBuffer: GPUBuffer;
  countBuffer: GPUBuffer;
  prefixSumBuffer: GPUBuffer;
  zeroBuffer: GPUBuffer;
  sortedBuffer: GPUBuffer;
  cellBindGroups: [GPUBindGroup, GPUBindGroup];
  sortBindGroup: GPUBindGroup;
  valenceBindGroups: [GPUBindGroup, GPUBindGroup];
  bioBindGroups: [GPUBindGroup, GPUBindGroup];
  simBindGroups: [GPUBindGroup, GPUBindGroup];
  prefixSumModule: InclusivePrefixSum;
}

let workgroupSize = 128;

let cellPipeline: GPUComputePipeline | undefined;
let sortPipeline: GPUComputePipeline | undefined;
let valencePipeline: GPUComputePipeline | undefined;
let bioPipeline: GPUComputePipeline | undefined;
let simPipeline: GPUComputePipeline | undefined;

let worldState: PrefixWorldState | undefined;

function destroyState() {
  if (!worldState) return;
  worldState.prefixSumModule.destroy();
  worldState.cellBuffer.destroy();
  worldState.sortedBuffer.destroy();
  worldState.countBuffer.destroy();
  worldState.prefixSumBuffer.destroy();
  worldState.zeroBuffer.destroy();
  worldState = undefined;
}

export function setup(device: GPUDevice) {
  workgroupSize = Math.min(128, device.limits.maxComputeWorkgroupSizeX);

  setupTimestamp(device, 'cell');
  cellPipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ code: cellComputeShader(workgroupSize) }),
      entryPoint: 'main',
    },
  });

  setupTimestamp(device, 'sort');
  sortPipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ code: sortComputeShader(workgroupSize) }),
      entryPoint: 'main',
    },
  });

  setupTimestamp(device, 'psValence');
  valencePipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ code: valenceComputeShader(workgroupSize) }),
      entryPoint: 'main',
    },
  });

  bioPipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ code: bioStepComputeShader(workgroupSize) }),
      entryPoint: 'main',
    },
  });

  setupTimestamp(device, 'countSim');
  simPipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ code: simComputeShader(workgroupSize) }),
      entryPoint: 'main',
    },
  });
}

export function start(device: GPUDevice, refs: UnifiedEngineRefs) {
  if (!cellPipeline || !sortPipeline || !valencePipeline || !bioPipeline || !simPipeline) {
    return;
  }

  destroyState();

  const { particleAmt, cellAmt } = refs;
  const cellBuffer = device.createBuffer({
    size: particleAmt * 12 * Float32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    label: 'psCellBuffer',
  });
  const sortedBuffer = device.createBuffer({
    size: particleAmt * 12 * Float32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    label: 'psSortedBuffer',
  });
  const countBuffer = device.createBuffer({
    size: cellAmt * Uint32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    label: 'psCountBuffer',
  });
  const prefixSumBuffer = device.createBuffer({
    size: cellAmt * Uint32Array.BYTES_PER_ELEMENT + 2 * Uint32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    label: 'psPrefixSumBuffer',
  });
  const zeroBuffer = device.createBuffer({
    size: cellAmt * Uint32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    label: 'psZeroBuffer',
  });
  device.queue.writeBuffer(zeroBuffer, 0, new Uint32Array(cellAmt));

  const sortBindGroup = device.createBindGroup({
    layout: sortPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: cellBuffer } },
      { binding: 1, resource: { buffer: sortedBuffer } },
      { binding: 2, resource: { buffer: prefixSumBuffer } },
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
          { binding: 7, resource: { buffer: prefixSumBuffer } },
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
          { binding: 9, resource: { buffer: prefixSumBuffer } },
          { binding: 10, resource: refs.windTexView },
          { binding: 11, resource: refs.tempTexView },
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
          { binding: 3, resource: refs.windTexView },
          { binding: 4, resource: refs.oceanTexView },
          { binding: 5, resource: { buffer: refs.rulesMatrixBuffer } },
          { binding: 6, resource: { buffer: refs.particleBuffers[i] } },
          { binding: 7, resource: { buffer: refs.particleBuffers[1 - i] } },
          { binding: 8, resource: { buffer: refs.lifeStateBuffers[1 - i] } },
          { binding: 9, resource: { buffer: refs.valenceBuffer } },
          { binding: 10, resource: { buffer: refs.bioForceBuffer } },
          { binding: 11, resource: { buffer: sortedBuffer } },
          { binding: 12, resource: { buffer: prefixSumBuffer } },
          { binding: 13, resource: refs.tempTexView },
        ],
      }),
    );
  }

  worldState = {
    cellBuffer,
    countBuffer,
    prefixSumBuffer,
    zeroBuffer,
    sortedBuffer,
    cellBindGroups: [cellGroups[0], cellGroups[1]],
    sortBindGroup,
    valenceBindGroups: [valenceGroups[0], valenceGroups[1]],
    bioBindGroups: [bioGroups[0], bioGroups[1]],
    simBindGroups: [simGroups[0], simGroups[1]],
    prefixSumModule: new InclusivePrefixSum(device, countBuffer, prefixSumBuffer, 2),
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
    !sortPipeline ||
    !valencePipeline ||
    !bioPipeline ||
    !simPipeline
  ) {
    return;
  }

  const wgCount = Math.ceil(particleAmt / workgroupSize);

  commandEncoder.copyBufferToBuffer(s.zeroBuffer, 0, s.countBuffer, 0, s.countBuffer.size);

  const cellPass = commandEncoder.beginComputePass(linkComputeTimestamp(device, 'cell'));
  cellPass.setPipeline(cellPipeline);
  cellPass.setBindGroup(0, s.cellBindGroups[alternate]);
  cellPass.dispatchWorkgroups(wgCount);
  cellPass.end();
  resolveTimestamp(commandEncoder, 'cell');

  s.prefixSumModule.run(commandEncoder, true);

  const sortPass = commandEncoder.beginComputePass(linkComputeTimestamp(device, 'sort'));
  sortPass.setPipeline(sortPipeline);
  sortPass.setBindGroup(0, s.sortBindGroup);
  sortPass.dispatchWorkgroups(wgCount);
  sortPass.end();
  resolveTimestamp(commandEncoder, 'sort');

  const valencePass = commandEncoder.beginComputePass(
    linkComputeTimestamp(device, 'psValence'),
  );
  valencePass.setPipeline(valencePipeline);
  valencePass.setBindGroup(0, s.valenceBindGroups[alternate]);
  valencePass.dispatchWorkgroups(wgCount);
  valencePass.end();
  resolveTimestamp(commandEncoder, 'psValence');

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
  const s = worldState;
  if (!s) return;
  readTimestamp('cell').then((time) => {
    params.cell = time;
  });
  s.prefixSumModule.updateDisplays(params);
  readTimestamp('sort').then((time) => {
    params.sort = time;
  });
  readTimestamp('psValence').then((time) => {
    params.valence = time;
  });
  readTimestamp('countSim').then((time) => {
    params.sim = time;
  });
}
