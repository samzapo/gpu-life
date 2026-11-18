import structsShader from '../shaders/structs.wgsl?raw';
import valenceSimShader from '../shaders/valenceSim.wgsl?raw';
import valenceShader from './valence.wgsl?raw';
import computeShader from './compute.wgsl?raw';
import {
  linkComputeTimestamp,
  readTimestamp,
  resolveTimestamp,
  setupTimestamp,
} from '../utils';

let valencePipeline: GPUComputePipeline | undefined;
let pipeline: GPUComputePipeline | undefined;
let valenceBindGroups: [GPUBindGroup, GPUBindGroup] | undefined;
let bindGroups: [GPUBindGroup, GPUBindGroup] | undefined;

const workgroupSize = 64;

export function setup(device: GPUDevice) {
  const valenceModule = device.createShaderModule({
    code: `${structsShader}\n${valenceSimShader}\n${valenceShader}`,
  });

  setupTimestamp(device, 'n2Valence');

  valencePipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: valenceModule,
      entryPoint: 'main',
    },
  });

  const module = device.createShaderModule({
    code: `${structsShader}\n${valenceSimShader}\n${computeShader}`,
  });

  setupTimestamp(device, 'compute');

  pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module,
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
) {
  if (!valencePipeline || !pipeline) return;
  const vGroups = [];
  const groups = [];
  for (let i = 0; i < 2; i++) {
    vGroups.push(
      device.createBindGroup({
        layout: valencePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 1, resource: { buffer: simBuffer } },
          { binding: 2, resource: { buffer: matrixBuffer } },
          { binding: 3, resource: { buffer: particleBuffers[i] } },
          { binding: 4, resource: { buffer: valenceBuffer } },
        ],
      }),
    );
    groups.push(
      device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uniformBuffer } },
          { binding: 1, resource: { buffer: simBuffer } },
          { binding: 2, resource: { buffer: matrixBuffer } },
          { binding: 3, resource: { buffer: particleBuffers[i] } },
          { binding: 4, resource: { buffer: particleBuffers[1 - i] } },
          { binding: 5, resource: { buffer: valenceBuffer } },
        ],
      }),
    );
  }
  valenceBindGroups = [vGroups[0], vGroups[1]];
  bindGroups = [groups[0], groups[1]];
}

export function tick(
  device: GPUDevice,
  commandEncoder: GPUCommandEncoder,
  alternate: number,
  particleAmt: number,
) {
  if (!valencePipeline || !pipeline || !valenceBindGroups || !bindGroups) return;

  const valencePass = commandEncoder.beginComputePass(
    linkComputeTimestamp(device, 'n2Valence'),
  );
  valencePass.setPipeline(valencePipeline);
  valencePass.setBindGroup(0, valenceBindGroups[alternate]);
  valencePass.dispatchWorkgroups(Math.ceil(particleAmt / workgroupSize));
  valencePass.end();
  resolveTimestamp(commandEncoder, 'n2Valence');

  const passEncoder = commandEncoder.beginComputePass(
    linkComputeTimestamp(device, 'compute'),
  );
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroups[alternate]);
  passEncoder.dispatchWorkgroups(Math.ceil(particleAmt / workgroupSize));
  passEncoder.end();
  resolveTimestamp(commandEncoder, 'compute');
}

export function updateDisplays(params: Record<string, number>) {
  readTimestamp('n2Valence').then((time) => {
    params.valence = time;
  });
  readTimestamp('compute').then((time) => {
    params.sim = time;
  });
}
