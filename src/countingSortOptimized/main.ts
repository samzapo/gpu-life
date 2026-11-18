import {
  linkComputeTimestamp,
  readTimestamp,
  resolveTimestamp,
  setupTimestamp,
} from '../utils';

import { cellComputeShader } from './computeShaders/cell';
import { sortComputeShader } from './computeShaders/sort';
import { valenceComputeShader } from './computeShaders/valence';
import { simComputeShader } from './computeShaders/sim';
import { oceanSimComputeShader } from './computeShaders/oceanSim';
import { InclusivePrefixSum } from './InclusivePrefixSum';

export type SimSlot = 'air' | 'ocean';

interface PrefixSlotState {
  cellBuffer: GPUBuffer;
  countBuffer: GPUBuffer;
  prefixSumBuffer: GPUBuffer;
  zeroBuffer: GPUBuffer;
  sortedBuffer: GPUBuffer;
  cellBindGroups: [GPUBindGroup, GPUBindGroup];
  sortBindGroup: GPUBindGroup;
  valenceBindGroups?: [GPUBindGroup, GPUBindGroup];
  simBindGroups: [GPUBindGroup, GPUBindGroup];
  prefixSumModule: InclusivePrefixSum;
}

let workgroupSize = 128;

let cellPipeline: GPUComputePipeline | undefined;
let sortPipeline: GPUComputePipeline | undefined;
let valencePipeline: GPUComputePipeline | undefined;
let simPipeline: GPUComputePipeline | undefined;
let oceanSimPipeline: GPUComputePipeline | undefined;

const slots: Partial<Record<SimSlot, PrefixSlotState>> = {};

function destroySlot(slot: SimSlot) {
  const s = slots[slot];
  if (!s) return;
  s.prefixSumModule.destroy();
  s.cellBuffer.destroy();
  s.sortedBuffer.destroy();
  s.countBuffer.destroy();
  s.prefixSumBuffer.destroy();
  s.zeroBuffer.destroy();
  delete slots[slot];
}

export function setup(device: GPUDevice) {
  workgroupSize = device.limits.maxComputeWorkgroupSizeX;

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

  setupTimestamp(device, 'countSim');
  simPipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ code: simComputeShader(workgroupSize) }),
      entryPoint: 'main',
    },
  });

  oceanSimPipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ code: oceanSimComputeShader(workgroupSize) }),
      entryPoint: 'main',
    },
  });
}

export function start(
  device: GPUDevice,
  uniformBuffer: GPUBuffer,
  simBuffer: GPUBuffer,
  matrixBuffer: GPUBuffer,
  valenceBuffer: GPUBuffer,
  particleBuffers: [GPUBuffer, GPUBuffer],
  particleAmt: number,
  cellAmt: number,
  fieldTexView: GPUTextureView,
  slot: SimSlot = 'air',
) {
  if (!cellPipeline || !sortPipeline || !valencePipeline || !simPipeline || !oceanSimPipeline) {
    return;
  }

  destroySlot(slot);

  const cellBuffer = device.createBuffer({
    size: particleAmt * 12 * Float32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    label: `${slot}CellBuffer`,
  });
  const sortedBuffer = device.createBuffer({
    size: particleAmt * 12 * Float32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    label: `${slot}SortedBuffer`,
  });
  const countBuffer = device.createBuffer({
    size: cellAmt * Uint32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    label: `${slot}CountBuffer`,
  });
  const prefixSumBuffer = device.createBuffer({
    size: cellAmt * Uint32Array.BYTES_PER_ELEMENT + 2 * Uint32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    label: `${slot}PrefixSumBuffer`,
  });
  const zeroBuffer = device.createBuffer({
    size: cellAmt * Uint32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    label: `${slot}ZeroBuffer`,
  });

  device.queue.writeBuffer(zeroBuffer, 0, new Uint32Array(cellAmt));

  const cellGroups = [];
  for (let i = 0; i < 2; i++) {
    cellGroups.push(
      device.createBindGroup({
        layout: cellPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: simBuffer } },
          { binding: 1, resource: { buffer: particleBuffers[i] } },
          { binding: 2, resource: { buffer: cellBuffer } },
          { binding: 3, resource: { buffer: countBuffer } },
        ],
      }),
    );
  }

  const sortBindGroup = device.createBindGroup({
    layout: sortPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: cellBuffer } },
      { binding: 1, resource: { buffer: sortedBuffer } },
      { binding: 2, resource: { buffer: prefixSumBuffer } },
    ],
  });

  let valenceBindGroups: [GPUBindGroup, GPUBindGroup] | undefined;
  const simGroups: GPUBindGroup[] = [];

  if (slot === 'air') {
    const vGroups: GPUBindGroup[] = [];
    for (let i = 0; i < 2; i++) {
      vGroups.push(
        device.createBindGroup({
          layout: valencePipeline.getBindGroupLayout(0),
          entries: [
            { binding: 1, resource: { buffer: simBuffer } },
            { binding: 2, resource: { buffer: matrixBuffer } },
            { binding: 3, resource: { buffer: particleBuffers[i] } },
            { binding: 4, resource: { buffer: valenceBuffer } },
            { binding: 5, resource: { buffer: sortedBuffer } },
            { binding: 6, resource: { buffer: countBuffer } },
            { binding: 7, resource: { buffer: prefixSumBuffer } },
          ],
        }),
      );
      simGroups.push(
        device.createBindGroup({
          layout: simPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: uniformBuffer } },
            { binding: 1, resource: { buffer: simBuffer } },
            { binding: 2, resource: { buffer: matrixBuffer } },
            { binding: 3, resource: { buffer: particleBuffers[i] } },
            { binding: 4, resource: { buffer: particleBuffers[1 - i] } },
            { binding: 5, resource: { buffer: sortedBuffer } },
            { binding: 6, resource: { buffer: countBuffer } },
            { binding: 7, resource: { buffer: prefixSumBuffer } },
            { binding: 8, resource: { buffer: valenceBuffer } },
            { binding: 9, resource: fieldTexView },
          ],
        }),
      );
    }
    valenceBindGroups = [vGroups[0], vGroups[1]];
  } else {
    for (let i = 0; i < 2; i++) {
      simGroups.push(
        device.createBindGroup({
          layout: oceanSimPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: uniformBuffer } },
            { binding: 1, resource: { buffer: simBuffer } },
            { binding: 2, resource: { buffer: matrixBuffer } },
            { binding: 3, resource: { buffer: particleBuffers[i] } },
            { binding: 4, resource: { buffer: particleBuffers[1 - i] } },
            { binding: 5, resource: { buffer: sortedBuffer } },
            { binding: 6, resource: { buffer: countBuffer } },
            { binding: 7, resource: { buffer: prefixSumBuffer } },
            { binding: 8, resource: fieldTexView },
          ],
        }),
      );
    }
  }

  slots[slot] = {
    cellBuffer,
    countBuffer,
    prefixSumBuffer,
    zeroBuffer,
    sortedBuffer,
    cellBindGroups: [cellGroups[0], cellGroups[1]],
    sortBindGroup,
    valenceBindGroups,
    simBindGroups: [simGroups[0], simGroups[1]],
    prefixSumModule: new InclusivePrefixSum(device, countBuffer, prefixSumBuffer, 2),
  };
}

export function tick(
  device: GPUDevice,
  commandEncoder: GPUCommandEncoder,
  alternate: number,
  particleAmt: number,
  slot: SimSlot = 'air',
  profile: boolean = slot === 'air',
) {
  const s = slots[slot];
  if (!s || !cellPipeline || !sortPipeline || !valencePipeline || !simPipeline || !oceanSimPipeline) {
    return;
  }

  commandEncoder.copyBufferToBuffer(s.zeroBuffer, 0, s.countBuffer, 0, s.countBuffer.size);

  const cellPass = commandEncoder.beginComputePass(
    profile ? linkComputeTimestamp(device, 'cell') : {},
  );
  cellPass.setPipeline(cellPipeline);
  cellPass.setBindGroup(0, s.cellBindGroups[alternate]);
  cellPass.dispatchWorkgroups(Math.ceil(particleAmt / workgroupSize));
  cellPass.end();
  if (profile) resolveTimestamp(commandEncoder, 'cell');

  s.prefixSumModule.run(commandEncoder, profile);

  const sortPass = commandEncoder.beginComputePass(
    profile ? linkComputeTimestamp(device, 'sort') : {},
  );
  sortPass.setPipeline(sortPipeline);
  sortPass.setBindGroup(0, s.sortBindGroup);
  sortPass.dispatchWorkgroups(Math.ceil(particleAmt / workgroupSize));
  sortPass.end();
  if (profile) resolveTimestamp(commandEncoder, 'sort');

  if (slot === 'air' && s.valenceBindGroups) {
    const valencePass = commandEncoder.beginComputePass(
      profile ? linkComputeTimestamp(device, 'psValence') : {},
    );
    valencePass.setPipeline(valencePipeline);
    valencePass.setBindGroup(0, s.valenceBindGroups[alternate]);
    valencePass.dispatchWorkgroups(Math.ceil(particleAmt / workgroupSize));
    valencePass.end();
    if (profile) resolveTimestamp(commandEncoder, 'psValence');

    const simPass = commandEncoder.beginComputePass(
      profile ? linkComputeTimestamp(device, 'countSim') : {},
    );
    simPass.setPipeline(simPipeline);
    simPass.setBindGroup(0, s.simBindGroups[alternate]);
    simPass.dispatchWorkgroups(Math.ceil(particleAmt / workgroupSize));
    simPass.end();
    if (profile) resolveTimestamp(commandEncoder, 'countSim');
  } else {
    const oceanPass = commandEncoder.beginComputePass(
      profile ? linkComputeTimestamp(device, 'countSim') : {},
    );
    oceanPass.setPipeline(oceanSimPipeline);
    oceanPass.setBindGroup(0, s.simBindGroups[alternate]);
    oceanPass.dispatchWorkgroups(Math.ceil(particleAmt / workgroupSize));
    oceanPass.end();
    if (profile) resolveTimestamp(commandEncoder, 'countSim');
  }
}

export function updateDisplays(params: Record<string, number>) {
  const s = slots.air ?? slots.ocean;
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
