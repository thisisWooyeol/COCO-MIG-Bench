# Evaluation Results on COCO-MIG Benchmark

Following COCO-MIG benchmark protocol (described in [MIGC paper](https://arxiv.org/abs/2402.05408)), we use 8 seeds (42, 37, 519, 609, 123, 401, 780, 0) to generate images for each prompt.
There are 160 prompts for each level, so total of 6400 images are generated and evaluated.
Each level of the benchmark is evaluated separately, and the results are averaged across all seeds.

We additionally reported time and VRAM usage for each method in float32 precision. For time, we measured the mean value for 1 iteration of all dataset. For VRAM usage, we measured the mean value of peak VRAM usage during the 1 iteration of all dataset.

<table style="text-align: center;">
  <thead>
    <tr>
      <th rowspan="2" style="text-align: center;">Method</th>
      <th rowspan="2" style="text-align: center;">Backbone</th>
      <th colspan="6" style="text-align: center;">Instance Attribute Success Ratio(%)↑</th>
      <th colspan="6" style="text-align: center;">Mean Intersection over Union(%)↑</th>
      <th colspan="2" style="text-align: center;">Image Text Consistency↑</th>
      <th rowspan="2" style="text-align: center;">Time(s)↓</th>
      <th rowspan="2" style="text-align: center;">VRAM (GB)↓</th>
    </tr>
    <tr>
      <th>L2</th>
      <th>L3</th>
      <th>L4</th>
      <th>L5</th>
      <th>L6</th>
      <th>Avg</th>
      <th>L2</th>
      <th>L3</th>
      <th>L4</th>
      <th>L5</th>
      <th>L6</th>
      <th>Avg</th>
      <th>CLIP</th>
      <th>Local CLIP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a>SD1.5</a></td>
      <td>SD1.5</td>
      <td>5.586</td>
      <td>4.792</td>
      <td>2.832</td>
      <td>2.406</td>
      <td>2.214</td>
      <td>3.109</td>
      <td>18.83</td>
      <td>17.43</td>
      <td>14.95</td>
      <td>13.93</td>
      <td>15.94</td>
      <td>15.75</td>
      <td>24.64</td>
      <td>18.36</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td><a>SDXL</a></td>
      <td>SDXL</td>
      <td>5.586</td>
      <td>4.479</td>
      <td>2.813</td>
      <td>2.141</td>
      <td>2.799</td>
      <td>3.168</td>
      <td>19.78</td>
      <td>18.54</td>
      <td>16.67</td>
      <td>15.72</td>
      <td>18.42</td>
      <td>17.55</td>
      <td>25.71</td>
      <td>18.63</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td><a>SD3.5-M</a></td>
      <td>SD3.5-M</td>
      <td>8.047</td>
      <td>8.464</td>
      <td>6.074</td>
      <td>5.016</td>
      <td>4.401</td>
      <td>5.863</td>
      <td>21.57</td>
      <td>21.37</td>
      <td>18.98</td>
      <td>17.39</td>
      <td>17.80</td>
      <td>18.85</td>
      <td>26.41</td>
      <td>18.77</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td><a>Flux.1-dev</a></td>
      <td>Flux.1-dev</td>
      <td>8.828</td>
      <td>7.500</td>
      <td>4.863</td>
      <td>4.531</td>
      <td>3.216</td>
      <td>5.078</td>
      <td>22.00</td>
      <td>20.93</td>
      <td>17.75</td>
      <td>16.77</td>
      <td>16.49</td>
      <td>18.03</td>
      <td>26.17</td>
      <td>18.56</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td colspan="18" style="text-align: center; font-weight: bold;">Adapter rendering methods</td>
    </tr>
    <tr>
      <td><a href="https://github.com/gligen/GLIGEN">GLIGEN</a></td>
      <td>SD1.4</td>
      <td>38.36</td>
      <td>32.79</td>
      <td>28.67</td>
      <td>25.02</td>
      <td>26.98</td>
      <td>28.84</td>
      <td>33.96</td>
      <td>29.58</td>
      <td>25.95</td>
      <td>23.88</td>
      <td>24.93</td>
      <td>26.47</td>
      <td>24.91</td>
      <td>20.78</td>
      <td><b>7.2911</b></td>
      <td><u>5.7944</u></td>
    </tr>
    <tr>
      <td><a href="https://github.com/frank-xwang/InstanceDiffusion">InstanceDiffusion</a></td>
      <td>SD1.5</td>
      <td><b>68.24</b></td>
      <td><u>60.47</u></td>
      <td><u>59.88</u></td>
      <td><u>53.92</u></td>
      <td><u>57.14</u></td>
      <td><u>58.49</u></td>
      <td><b>62.67</b></td>
      <td><b>55.75</b></td>
      <td><b>54.15</b></td>
      <td><u>49.02</u></td>
      <td><u>51.34</u></td>
      <td><b>53.12</b></td>
      <td><b>25.97</b></td>
      <td><b>21.90</b></td>
      <td>26.672</td>
      <td>6.3914</td>
    </tr>
    <tr>
      <td><a href="https://github.com/limuloo/MIGC">MIGC</a></td>
      <td>SD1.4</td>
      <td><u>66.37</u></td>
      <td><b>63.10</b></td>
      <td><b>61.27</b></td>
      <td><b>57.25</b></td>
      <td><b>59.13</b></td>
      <td><b>60.41</b></td>
      <td><u>57.02</u></td>
      <td><u>54.47</u></td>
      <td><u>52.48</u></td>
      <td><b>49.49</b></td>
      <td><b>51.38</b></td>
      <td><u>52.16</u></td>
      <td><u>25.39</u></td>
      <td><u>21.42</u></td>
      <td><u>7.3080</u></td>
      <td><b>5.2236</b></td>
    </tr>
    <tr>
      <td><a href="https://github.com/limuloo/3DIS/">3DIS</a></td>
      <td>SD1.5</td>
      <td>58.09</td>
      <td>51.48</td>
      <td>46.15</td>
      <td>40.39</td>
      <td>41.22</td>
      <td>45.23</td>
      <td>52.76</td>
      <td>46.92</td>
      <td>42.46</td>
      <td>38.16</td>
      <td>38.47</td>
      <td>41.89</td>
      <td>24.02</td>
      <td>21.24</td>
      <td>10.991</td>
      <td>7.5521</td>
    </tr>
  </tbody>
</table>

## Reference table from 3DIS

<table style="text-align: center;">
  <thead>
    <tr>
      <th rowspan="2" style="text-align: center;">Method</th>
      <th colspan="6" style="text-align: center;">Instance Attribute Success Ratio(%)↑</th>
      <th colspan="6" style="text-align: center;">Mean Intersection over Union(%)↑</th>
      <th rowspan="2" style="text-align: center;">Venue</th>
    </tr>
    <tr>
      <th>L2</th>
      <th>L3</th>
      <th>L4</th>
      <th>L5</th>
      <th>L6</th>
      <th>Avg</th>
      <th>L2</th>
      <th>L3</th>
      <th>L4</th>
      <th>L5</th>
      <th>L6</th>
      <th>Avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="16" style="text-align: center; font-weight: bold;">Adapter rendering methods</td>
    </tr>
    <tr>
      <td><a href="https://github.com/gligen/GLIGEN">GLIGEN</a></td>
      <td>41.3</td>
      <td>33.8</td>
      <td>31.8</td>
      <td>27.0</td>
      <td>29.5</td>
      <td>31.3</td>
      <td>33.7</td>
      <td>27.6</td>
      <td>25.5</td>
      <td>21.9</td>
      <td>23.6</td>
      <td>25.2</td>
      <td>CVPR2023</td>
    </tr>
    <tr>
      <td><a href="https://github.com/frank-xwang/InstanceDiffusion">InstanceDiffusion</a></td>
      <td>61.0</td>
      <td>52.8</td>
      <td>52.4</td>
      <td>45.2</td>
      <td>48.7</td>
      <td>50.5</td>
      <td>53.8</td>
      <td>45.8</td>
      <td>44.9</td>
      <td>37.7</td>
      <td>40.6</td>
      <td>43.0</td>
      <td>CVPR2024</td>
    </tr>
    <tr>
      <td><a href="https://github.com/limuloo/MIGC">MIGC</a></td>
      <td><b>74.8</b></td>
      <td><b>66.2</b></td>
      <td><b>67.4</b></td>
      <td><b>65.3</b></td>
      <td><b>66.1</b></td>
      <td><b>67.1</b></td>
      <td><b>63.0</b></td>
      <td><b>54.7</b></td>
      <td><b>55.3</b></td>
      <td><b>52.4</b></td>
      <td><b>53.2</b></td>
      <td><b>54.7</b></td>
      <td>CVPR2024</td>
    </tr>
    <tr>
      <td colspan="16" style="text-align: center; font-weight: bold;">training-free rendering</td>
    </tr>
    <tr>
      <td><a href="https://github.com/silent-chen/layout-guidance">TFLCG</a></td>
      <td>17.2</td>
      <td>13.5</td>
      <td>7.9</td>
      <td>6.1</td>
      <td>4.5</td>
      <td>8.3</td>
      <td>10.9</td>
      <td>8.7</td>
      <td>5.1</td>
      <td>3.9</td>
      <td>2.8</td>
      <td>5.3</td>
      <td>WACV2024</td>
    </tr>
    <tr>
      <td><a href="https://github.com/showlab/BoxDiff">Box-Diffusion</a></td>
      <td>28.4</td>
      <td>21.4</td>
      <td>14.0</td>
      <td>11.9</td>
      <td>12.8</td>
      <td>15.7</td>
      <td>19.1</td>
      <td>14.6</td>
      <td>9.4</td>
      <td>7.9</td>
      <td>8.5</td>
      <td>10.6</td>
      <td>ICCV2023</td>
    </tr>
    <tr>
      <td><a href="https://github.com/omerbt/MultiDiffusion">Multi Diffusion</a></td>
      <td>30.6</td>
      <td>25.3</td>
      <td>24.5</td>
      <td>18.3</td>
      <td>19.8</td>
      <td>22.3</td>
      <td>21.9</td>
      <td>18.1</td>
      <td>17.3</td>
      <td>12.9</td>
      <td>13.9</td>
      <td>15.8</td>
      <td>ICML2023</td>
    </tr>
    <tr>
      <td>3DIS (SD1.5)</td>
      <td>65.9</td>
      <td>56.1</td>
      <td>55.3</td>
      <td>45.3</td>
      <td>47.6</td>
      <td>53.0</td>
      <td>56.8</td>
      <td>48.4</td>
      <td>49.4</td>
      <td>40.2</td>
      <td>41.7</td>
      <td>44.7</td>
      <td>ICLR2025</td>
    </tr>
    <tr>
      <td>3DIS (SD2.1)</td>
      <td>66.1</td>
      <td>57.5</td>
      <td>55.1</td>
      <td>51.7</td>
      <td>52.9</td>
      <td>54.7</td>
      <td>57.1</td>
      <td>48.6</td>
      <td>46.8</td>
      <td>42.9</td>
      <td>43.4</td>
      <td>45.7</td>
      <td>ICLR2025</td>
    </tr>
    <tr>
      <td>3DIS (SDXL)</td>
      <td>66.1</td>
      <td>59.3</td>
      <td>56.2</td>
      <td>51.7</td>
      <td>54.1</td>
      <td>56.0</td>
      <td>57.0</td>
      <td>50.0</td>
      <td>47.8</td>
      <td>43.1</td>
      <td>44.6</td>
      <td>47.0</td>
      <td>ICLR2025</td>
    </tr>
    <tr>
      <td>vs. MultiDiff</td>
      <td><b style="color: green;">+35</b></td>
      <td><b style="color: green;">+34</b></td>
      <td><b style="color: green;">+31</b></td>
      <td><b style="color: green;">+33</b></td>
      <td><b style="color: green;">+34</b></td>
      <td><b style="color: green;">+33</b></td>
      <td><b style="color: green;">+35</b></td>
      <td><b style="color: green;">+31</b></td>
      <td><b style="color: green;">+30</b></td>
      <td><b style="color: green;">+30</b></td>
      <td><b style="color: green;">+30</b></td>
      <td><b style="color: green;">+31</b></td>
      <td>-</td>
    </tr>
    <tr>
      <td colspan="17" style="text-align: center; font-weight: bold;">rendering w/ off-the-shelf adapters</td>
    </tr>
    <tr>
      <td>3DIS+GLIGEN</td>
      <td>49.4</td>
      <td>39.7</td>
      <td>34.5</td>
      <td>29.6</td>
      <td>29.9</td>
      <td>34.1</td>
      <td>43.0</td>
      <td>33.8</td>
      <td>29.2</td>
      <td>24.6</td>
      <td>24.5</td>
      <td>28.8</td>
      <td>-</td>
    </tr>
    <tr>
      <td>vs. GLIGEN</td>
      <td><b style="color: green;">+8.1</b></td>
      <td><b style="color: green;">+5.9</b></td>
      <td><b style="color: green;">+2.7</b></td>
      <td><b style="color: green;">+2.6</b></td>
      <td><b style="color: green;">+0.4</b></td>
      <td><b style="color: green;">+2.8</b></td>
      <td><b style="color: green;">+9.3</b></td>
      <td><b style="color: green;">+6.2</b></td>
      <td><b style="color: green;">+3.7</b></td>
      <td><b style="color: green;">+2.7</b></td>
      <td><b style="color: green;">+0.9</b></td>
      <td><b style="color: green;">+3.6</b></td>
      <td>-</td>
    </tr>
    <tr>
      <td>3DIS+MIGC</td>
      <td><b>76.8</b></td>
      <td><b>70.2</b></td>
      <td><b>72.3</b></td>
      <td><b>66.4</b></td>
      <td><b>68.0</b></td>
      <td><b>69.7</b></td>
      <td><b>68.0</b></td>
      <td><b>60.7</b></td>
      <td><b>62.0</b></td>
      <td><b>55.8</b></td>
      <td><b>57.3</b></td>
      <td><b>59.5</b></td>
      <td>-</td>
    </tr>
    <tr>
      <td>vs. MIGC</td>
      <td><b style="color: green;">+2.0</b></td>
      <td><b style="color: green;">+4.0</b></td>
      <td><b style="color: green;">+4.9</b></td>
      <td><b style="color: green;">+1.1</b></td>
      <td><b style="color: green;">+1.9</b></td>
      <td><b style="color: green;">+2.6</b></td>
      <td><b style="color: green;">+5.0</b></td>
      <td><b style="color: green;">+6.0</b></td>
      <td><b style="color: green;">+6.7</b></td>
      <td><b style="color: green;">+3.4</b></td>
      <td><b style="color: green;">+4.1</b></td>
      <td><b style="color: green;">+4.8</b></td>
      <td>-</td>
    </tr>
  </tbody>
</table>

## Detailed inference efficiency

Here are the detailed inference efficiency metrics by the level of prompts, including time and VRAM usage.

<table style="text-align: center;">
  <thead>
    <tr>
      <th rowspan="2" style="text-align: center;">Method</th>
      <th rowspan="2" style="text-align: center;">Backbone</th>
      <th colspan="6" style="text-align: center;">Time(s)↓</th>
      <th colspan="6" style="text-align: center;">VRAM (GB)↓</th>
    </tr>
    <tr>
      <th>L2</th>
      <th>L3</th>
      <th>L4</th>
      <th>L5</th>
      <th>L6</th>
      <th>Avg</th>
      <th>L2</th>
      <th>L3</th>
      <th>L4</th>
      <th>L5</th>
      <th>L6</th>
      <th>Avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="14" style="text-align: center; font-weight: bold;">Adapter rendering methods</td>
    </tr>
    <tr>
      <td><a href="https://github.com/gligen/GLIGEN">GLIGEN</a></td>
      <td>SD1.4</td>
      <td>7.2846</td>
      <td>7.2941</td>
      <td>7.2923</td>
      <td>7.2928</td>
      <td>7.2915</td>
      <td>7.2911</td>
      <td>5.7944</td>
      <td>5.7944</td>
      <td>5.7944</td>
      <td>5.7944</td>
      <td>5.7944</td>
      <td>5.7944</td>
    </tr>
    <tr>
      <td><a href="https://github.com/frank-xwang/InstanceDiffusion">InstanceDiffusion</a></td>
      <td>SD1.5</td>
      <td>18.655</td>
      <td>22.672</td>
      <td>26.674</td>
      <td>30.711</td>
      <td>34.647</td>
      <td>26.672</td>
      <td>6.3909</td>
      <td>6.3912</td>
      <td>6.3914</td>
      <td>6.3916</td>
      <td>6.3919</td>
      <td>6.3914</td>
    </tr>
    <tr>
      <td><a href="https://github.com/limuloo/MIGC">MIGC</a></td>
      <td>SD1.4</td>
      <td>7.2752</td>
      <td>7.2926</td>
      <td>7.3056</td>
      <td>7.3190</td>
      <td>7.3478</td>
      <td>7.3080</td>
      <td>5.2230</td>
      <td>5.2233</td>
      <td>5.2235</td>
      <td>5.2239</td>
      <td>5.2241</td>
      <td>5.2236</td>
    </tr>
    <tr>
      <td><a href="https://github.com/limuloo/3DIS/">3DIS</a></td>
      <td>SD1.5</td>
      <td>10.560</td>
      <td>10.844</td>
      <td>11.032</td>
      <td>11.141</td>
      <td>11.376</td>
      <td>10.991</td>
      <td>7.5515</td>
      <td>7.5522</td>
      <td>7.5519</td>
      <td>7.5523</td>
      <td>7.5526</td>
      <td>7.5521</td>
    </tr>
  </tbody>
</table>
