import structsShader from '../shaders/structs.wgsl?raw';
import projectionShader from '../shaders/projection.wgsl?raw';
import bioChemShader from '../shaders/bioChem.wgsl?raw';
import valenceSimShader from '../shaders/valenceSim.wgsl?raw';
import oceanSimHelperShader from '../shaders/oceanSim.wgsl?raw';
import constructShader from './construct.wgsl?raw';
import valenceShader from './valence.wgsl?raw';
import simShader from './sim.wgsl?raw';
import oceanSimShader from './oceanSim.wgsl?raw';
import bioAirShader from './bioAir.wgsl?raw';
import bioOceanShader from './bioOcean.wgsl?raw';
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
  matrixBuffer: GPUBuffer;
  valenceBuffer: GPUBuffer;
  particleBuffers: [GPUBuffer, GPUBuffer];
  constructBindGroups: [GPUBindGroup, GPUBindGroup];
  valenceBindGroups?: [GPUBindGroup, GPUBindGroup];
  simBindGroups: [GPUBindGroup, GPUBindGroup];
}

interface BioRefs {
  bioUniformBuffer: GPUBuffer;
  airBioReactBuffer: GPUBuffer;
  oceanSteerBuffer: GPUBuffer;
}

let constructPipeline: GPUComputePipeline | undefined;
let valencePipeline: GPUComputePipeline | undefined;
let simPipeline: GPUComputePipeline | undefined;
let oceanSimPipeline: GPUComputePipeline | undefined;
let bioAirPipeline: GPUComputePipeline | undefined;
let bioOceanPipeline: GPUComputePipeline | undefined;

const slots: Partial<Record<SimSlot, LinkedListSlotState>> = {};
let bioRefs: BioRefs | undefined;
let bioAirBindGroups: [GPUBindGroup, GPUBindGroup] | undefined;
let bioOceanBindGroups: [GPUBindGroup, GPUBindGroup] | undefined;

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
        code: `${structsShader}\n${projectionShader}\n${bioChemShader}\n${valenceSimShader}\n${valenceShader}`,
      }),
      entryPoint: 'main',
    },
  });

  setupTimestamp(device, 'sim');
  simPipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: device.createShaderModule({
        code: `${structsShader}\n${projectionShader}\n${bioChemShader}\n${valenceSimShader}\n${simShader}`,
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

  bioAirPipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: device.createShaderModule({
        code: `${structsShader}\n${projectionShader}\n${bioChemShader}\n${bioAirShader}`,
      }),
      entryPoint: 'main',
    },
  });

  bioOceanPipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: device.createShaderModule({
        code: `${structsShader}\n${projectionShader}\n${bioChemShader}\n${bioOceanShader}`,
      }),
      entryPoint: 'main',
    },
  });
}

function rebuildBioBindGroups(device: GPUDevice) {
  const air = slots.air;
  const ocean = slots.ocean;
  if (!air || !ocean || !bioRefs || !bioAirPipeline || !bioOceanPipeline) {
    bioAirBindGroups = undefined;
    bioOceanBindGroups = undefined;
    return;
  }

  bioAirBindGroups = [0, 1].map((i) =>
    device.createBindGroup({
      layout: bioAirPipeline!.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: bioRefs!.bioUniformBuffer } },
        { binding: 1, resource: { buffer: air.matrixBuffer } },
        { binding: 2, resource: { buffer: air.particleBuffers[i] } },
        { binding: 3, resource: { buffer: air.valenceBuffer } },
        { binding: 4, resource: { buffer: ocean.headsBuffer } },
        { binding: 5, resource: { buffer: ocean.linkedListBuffer } },
        { binding: 6, resource: { buffer: ocean.valenceBuffer } },
        { binding: 7, resource: { buffer: bioRefs!.airBioReactBuffer } },
      ],
    }),
  ) as [GPUBindGroup, GPUBindGroup];

  bioOceanBindGroups = [0, 1].map((i) =>
    device.createBindGroup({
      layout: bioOceanPipeline!.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: bioRefs!.bioUniformBuffer } },
        { binding: 1, resource: { buffer: air.matrixBuffer } },
        { binding: 2, resource: { buffer: air.valenceBuffer } },
        { binding: 3, resource: { buffer: air.headsBuffer } },
        { binding: 4, resource: { buffer: air.linkedListBuffer } },
        { binding: 5, resource: { buffer: ocean.particleBuffers[i] } },
        { binding: 6, resource: { buffer: ocean.valenceBuffer } },
        { binding: 7, resource: { buffer: bioRefs!.oceanSteerBuffer } },
      ],
    }),
  ) as [GPUBindGroup, GPUBindGroup];
}

export function bindCrossBio(
  device: GPUDevice,
  bioUniformBuffer: GPUBuffer,
  airBioReactBuffer: GPUBuffer,
  oceanSteerBuffer: GPUBuffer,
) {
  bioRefs = { bioUniformBuffer, airBioReactBuffer, oceanSteerBuffer };
  rebuildBioBindGroups(device);
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
  bioSideBuffer?: GPUBuffer,
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
            { binding: 9, resource: { buffer: bioSideBuffer! } },
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
            { binding: 8, resource: { buffer: valenceBuffer } },
            { binding: 9, resource: { buffer: bioSideBuffer! } },
          ],
        }),
      );
    }
  }

  slots[slot] = {
    headsBuffer,
    headsInitBuffer,
    linkedListBuffer,
    matrixBuffer,
    valenceBuffer,
    particleBuffers,
    constructBindGroups: [constructGroups[0], constructGroups[1]],
    valenceBindGroups,
    simBindGroups: [simGroups[0], simGroups[1]],
  };

  rebuildBioBindGroups(device);
}

export function buildIndex(
  device: GPUDevice,
  commandEncoder: GPUCommandEncoder,
  alternate: number,
  particleAmt: number,
  slot: SimSlot = 'air',
  profile: boolean = slot === 'air',
) {
  const s = slots[slot];
  if (!s || !constructPipeline || !valencePipeline) {
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
  }
}

export function stepBio(
  commandEncoder: GPUCommandEncoder,
  airAlt: number,
  oceanAlt: number,
  airAmt: number,
  oceanAmt: number,
) {
  if (!bioAirPipeline || !bioOceanPipeline || !bioAirBindGroups || !bioOceanBindGroups) {
    return;
  }

  const airBioPass = commandEncoder.beginComputePass({});
  airBioPass.setPipeline(bioAirPipeline);
  airBioPass.setBindGroup(0, bioAirBindGroups[airAlt]);
  airBioPass.dispatchWorkgroups(Math.ceil(airAmt / workgroupSize));
  airBioPass.end();

  const oceanBioPass = commandEncoder.beginComputePass({});
  oceanBioPass.setPipeline(bioOceanPipeline);
  oceanBioPass.setBindGroup(0, bioOceanBindGroups[oceanAlt]);
  oceanBioPass.dispatchWorkgroups(Math.ceil(oceanAmt / workgroupSize));
  oceanBioPass.end();
}

export function stepSim(
  device: GPUDevice,
  commandEncoder: GPUCommandEncoder,
  alternate: number,
  particleAmt: number,
  slot: SimSlot = 'air',
  profile: boolean = slot === 'air',
) {
  const s = slots[slot];
  if (!s || !simPipeline || !oceanSimPipeline) {
    return;
  }

  if (slot === 'air') {
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

export function tick(
  device: GPUDevice,
  commandEncoder: GPUCommandEncoder,
  alternate: number,
  particleAmt: number,
  slot: SimSlot = 'air',
  profile: boolean = slot === 'air',
) {
  buildIndex(device, commandEncoder, alternate, particleAmt, slot, profile);
  stepSim(device, commandEncoder, alternate, particleAmt, slot, profile);
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
