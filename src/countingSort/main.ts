import {
  linkComputeTimestamp,
  readTimestamp,
  resolveTimestamp,
  setupTimestamp,
} from '../utils';

import structsShader from '../shaders/structs.wgsl?raw';
import projectionShader from '../shaders/projection.wgsl?raw';
import valenceSimShader from '../shaders/valenceSim.wgsl?raw';
import oceanSimHelperShader from '../shaders/oceanSim.wgsl?raw';
import cellShader from './cell.wgsl?raw';
import prefixShader from './prefix.wgsl?raw';
import sortShader from './sort.wgsl?raw';
import valenceShader from './valence.wgsl?raw';
import simShader from './sim.wgsl?raw';
import oceanSimShader from './oceanSim.wgsl?raw';

export type SimSlot = 'air' | 'ocean';

interface CountingSlotState {
  cellBuffer: GPUBuffer;
  countBuffer: GPUBuffer;
  zeroBuffer: GPUBuffer;
  sortedBuffer: GPUBuffer;
  indicesBuffer: GPUBuffer;
  cellBindGroups: [GPUBindGroup, GPUBindGroup];
  prefixBindGroup: GPUBindGroup;
  sortBindGroup: GPUBindGroup;
  valenceBindGroups?: [GPUBindGroup, GPUBindGroup];
  simBindGroups: [GPUBindGroup, GPUBindGroup];
}

const workgroupSize = 128;

let cellPipeline: GPUComputePipeline | undefined;
let prefixPipeline: GPUComputePipeline | undefined;
let sortPipeline: GPUComputePipeline | undefined;
let valencePipeline: GPUComputePipeline | undefined;
let simPipeline: GPUComputePipeline | undefined;
let oceanSimPipeline: GPUComputePipeline | undefined;

const slots: Partial<Record<SimSlot, CountingSlotState>> = {};

function destroySlot(slot: SimSlot) {
  const s = slots[slot];
  if (!s) return;
  s.cellBuffer.destroy();
  s.sortedBuffer.destroy();
  s.countBuffer.destroy();
  s.indicesBuffer.destroy();
  s.zeroBuffer.destroy();
  delete slots[slot];
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
        code: `${structsShader}\n${projectionShader}\n${valenceSimShader}\n${valenceShader}`,
      }),
      entryPoint: 'main',
    },
  });

  setupTimestamp(device, 'countSim');
  simPipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: device.createShaderModule({
        code: `${structsShader}\n${projectionShader}\n${valenceSimShader}\n${simShader}`,
      }),
      entryPoint: 'main',
    },
  });

  oceanSimPipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: device.createShaderModule({
        code: `${structsShader}\n${projectionShader}\n${oceanSimHelperShader}\n${oceanSimShader}`,
      }),
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
  if (
    !cellPipeline ||
    !prefixPipeline ||
    !sortPipeline ||
    !valencePipeline ||
    !simPipeline ||
    !oceanSimPipeline
  ) {
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
  const indicesBuffer = device.createBuffer({
    size: cellAmt * 2 * Uint32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    label: `${slot}IndicesBuffer`,
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
            { binding: 6, resource: { buffer: indicesBuffer } },
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
            { binding: 6, resource: { buffer: indicesBuffer } },
            { binding: 7, resource: { buffer: valenceBuffer } },
            { binding: 8, resource: fieldTexView },
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
            { binding: 6, resource: { buffer: indicesBuffer } },
            { binding: 7, resource: fieldTexView },
          ],
        }),
      );
    }
  }

  slots[slot] = {
    cellBuffer,
    countBuffer,
    zeroBuffer,
    sortedBuffer,
    indicesBuffer,
    cellBindGroups: [cellGroups[0], cellGroups[1]],
    prefixBindGroup,
    sortBindGroup,
    valenceBindGroups,
    simBindGroups: [simGroups[0], simGroups[1]],
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
  if (
    !s ||
    !cellPipeline ||
    !prefixPipeline ||
    !sortPipeline ||
    !valencePipeline ||
    !simPipeline ||
    !oceanSimPipeline
  ) {
    return;
  }

  commandEncoder.copyBufferToBuffer(s.zeroBuffer, s.countBuffer);

  const cellPass = commandEncoder.beginComputePass(
    profile ? linkComputeTimestamp(device, 'cell') : {},
  );
  cellPass.setPipeline(cellPipeline);
  cellPass.setBindGroup(0, s.cellBindGroups[alternate]);
  cellPass.dispatchWorkgroups(Math.ceil(particleAmt / workgroupSize));
  cellPass.end();
  if (profile) resolveTimestamp(commandEncoder, 'cell');

  const prefixPass = commandEncoder.beginComputePass(
    profile ? linkComputeTimestamp(device, 'prefix') : {},
  );
  prefixPass.setPipeline(prefixPipeline);
  prefixPass.setBindGroup(0, s.prefixBindGroup);
  prefixPass.dispatchWorkgroups(1);
  prefixPass.end();
  if (profile) resolveTimestamp(commandEncoder, 'prefix');

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
      profile ? linkComputeTimestamp(device, 'csValence') : {},
    );
    valencePass.setPipeline(valencePipeline);
    valencePass.setBindGroup(0, s.valenceBindGroups[alternate]);
    valencePass.dispatchWorkgroups(Math.ceil(particleAmt / workgroupSize));
    valencePass.end();
    if (profile) resolveTimestamp(commandEncoder, 'csValence');

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
