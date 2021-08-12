const lightCodeTheme = require('prism-react-renderer/themes/github');
const darkCodeTheme = require('prism-react-renderer/themes/dracula');

/** @type {import('@docusaurus/types').DocusaurusConfig} */
module.exports = {
  title: 'ml.wasm > supervised',
  tagline: 'Classification and Regression powered by WebAssembly',
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',
  // favicon: 'img/favicon.ico',
  organizationName: 'ml-wasm',
  projectName: 'supervised',
  url: 'https://ml-wasm.github.io',
  baseUrl: '/supervised/',
  trailingSlash: false,
  themeConfig: {
    navbar: {
      title: 'ml.wasm > supervised',
      // logo: {
      //  alt: 'My Site Logo',
      //  src: 'img/logo.svg',
      // },
      items: [
        {
          type: 'doc',
          docId: 'Getting Started',
          position: 'left',
          label: 'Docs',
        },
        {
          href: 'https://github.com/ml-wasm/supervised',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    prism: {
      theme: lightCodeTheme,
      darkTheme: darkCodeTheme,
    },
  },
  presets: [
    [
      '@docusaurus/preset-classic',
      {
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          routeBasePath: '/',
          editUrl:
            'https://github.com/ml-wasm/supervised/edit/main/docs/',
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      },
    ],
  ],
};
