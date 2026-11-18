import structsShader from '../shaders/structs.wgsl?raw';
import projectionShader from '../shaders/projection.wgsl?raw';
import valenceSimShader from '../shaders/valenceSim.wgsl?raw';
import oceanSimHelperShader from '../shaders/oceanSim.wgsl?raw';
import constructShader from './construct.wgsl?raw';
import valenceShader from './valence.wgsl?raw';
import simShader from './sim.wgsl?raw';
import oceanSimShader from './oceanSim.wgsl?raw';
import {
  linkComputeTimestamp,
  readTimestamp,
  resolveTimestamp,
  setupTimestamp,
} from '../utils';

export type SimSlot = 'air' | 'ocean';

interface LinkedListSlotState {
  headsBuffer: GPUBuffer;
  headsInitBuffer: GPUBuffer;
  linkedListBuffer: GPUBuffer;
  constructBindGroups: [GPUBindGroup, GPUBindGroup];
  valenceBindGroups?: [GPUBindGroup, GPUBindGroup];
  simBindGroups: [GPUBindGroup, GPUBindGroup];
}

let constructPipeline: GPUComputePipeline | undefined;
let valencePipeline: GPUComputePipeline | undefined;
let simPipeline: GPUComputePipeline | undefined;
let oceanSimPipeline: GPUComputePipeline | undefined;

const slots: Partial<Record<SimSlot, LinkedListSlotState>> = {};

const workgroupSize = 64;

function destroySlot(slot: SimSlot) {
  const s = slots[slot];
  if (!s) return;
  s.headsBuffer.destroy();
  s.headsInitBuffer.destroy();
  s.linkedListBuffer.destroy();
  delete slots[slot];
}

export function setup(device: GPUDevice) {
  setupTimestamp(device, 'construct');
  constructPipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ code: `${structsShader}\n${constructShader}` }),
      entryPoint: 'main',
    },
  });

  setupTimestamp(device, 'llValence');
  valencePipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: device.createShaderModule({
        code: `${structsShader}\n${projectionShader}\n${valenceSimShader}\n${valenceShader}`,
      }),
      entryPoint: 'main',
    },
  });

  setupTimestamp(device, 'sim');
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
  if (!constructPipeline || !valencePipeline || !simPipeline || !oceanSimPipeline) {
    return;
  }

  destroySlot(slot);

  const headsBuffer = device.createBuffer({
    size: (1 + cellAmt) * Uint32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    label: `${slot}HeadsBuffer`,
  });

  const headsInitBuffer = device.createBuffer({
    size: (1 + cellAmt) * Uint32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.COPY_SRC,
    mappedAtCreation: true,
    label: `${slot}HeadsInitBuffer`,
  });
  const initArr = new Uint32Array(headsInitBuffer.getMappedRange());
  initArr[0] = 0;
  for (let i = 1; i < initArr.length; ++i) {
    initArr[i] = 0xffffffff;
  }
  headsInitBuffer.unmap();

  const linkedListBuffer = device.createBuffer({
    size: 12 * Float32Array.BYTES_PER_ELEMENT * particleAmt,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    label: `${slot}LinkedListBuffer`,
  });

  const constructGroups = [];
  for (let i = 0; i < 2; i++) {
    constructGroups.push(
      device.createBindGroup({
        layout: constructPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: simBuffer } },
          { binding: 1, resource: { buffer: particleBuffers[i] } },
          { binding: 2, resource: { buffer: headsBuffer } },
          { binding: 3, resource: { buffer: linkedListBuffer } },
        ],
      }),
    );
  }

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
            { binding: 5, resource: { buffer: headsBuffer } },
            { binding: 6, resource: { buffer: linkedListBuffer } },
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
            { binding: 5, resource: { buffer: headsBuffer } },
            { binding: 6, resource: { buffer: linkedListBuffer } },
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
            { binding: 5, resource: { buffer: headsBuffer } },
            { binding: 6, resource: { buffer: linkedListBuffer } },
            { binding: 7, resource: fieldTexView },
          ],
        }),
      );
    }
  }

  slots[slot] = {
    headsBuffer,
    headsInitBuffer,
    linkedListBuffer,
    constructBindGroups: [constructGroups[0], constructGroups[1]],
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
  if (!s || !constructPipeline || !valencePipeline || !simPipeline || !oceanSimPipeline) {
    return;
  }

  commandEncoder.copyBufferToBuffer(s.headsInitBuffer, s.headsBuffer);

  const constructPass = commandEncoder.beginComputePass(
    profile ? linkComputeTimestamp(device, 'construct') : {},
  );
  constructPass.setPipeline(constructPipeline);
  constructPass.setBindGroup(0, s.constructBindGroups[alternate]);
  constructPass.dispatchWorkgroups(Math.ceil(particleAmt / workgroupSize));
  constructPass.end();
  if (profile) resolveTimestamp(commandEncoder, 'construct');

  if (slot === 'air' && s.valenceBindGroups) {
    const valencePass = commandEncoder.beginComputePass(
      profile ? linkComputeTimestamp(device, 'llValence') : {},
    );
    valencePass.setPipeline(valencePipeline);
    valencePass.setBindGroup(0, s.valenceBindGroups[alternate]);
    valencePass.dispatchWorkgroups(Math.ceil(particleAmt / workgroupSize));
    valencePass.end();
    if (profile) resolveTimestamp(commandEncoder, 'llValence');

    const simPass = commandEncoder.beginComputePass(
      profile ? linkComputeTimestamp(device, 'sim') : {},
    );
    simPass.setPipeline(simPipeline);
    simPass.setBindGroup(0, s.simBindGroups[alternate]);
    simPass.dispatchWorkgroups(Math.ceil(particleAmt / workgroupSize));
    simPass.end();
    if (profile) resolveTimestamp(commandEncoder, 'sim');
  } else {
    const oceanPass = commandEncoder.beginComputePass(
      profile ? linkComputeTimestamp(device, 'sim') : {},
    );
    oceanPass.setPipeline(oceanSimPipeline);
    oceanPass.setBindGroup(0, s.simBindGroups[alternate]);
    oceanPass.dispatchWorkgroups(Math.ceil(particleAmt / workgroupSize));
    oceanPass.end();
    if (profile) resolveTimestamp(commandEncoder, 'sim');
  }
}

export function updateDisplays(params: Record<string, number>) {
  readTimestamp('construct').then((time) => {
    params.construct = time;
  });
  readTimestamp('llValence').then((time) => {
    params.valence = time;
  });
  readTimestamp('sim').then((time) => {
    params.sim = time;
  });
}
