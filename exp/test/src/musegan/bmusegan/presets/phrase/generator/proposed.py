"""Network architecture of the proposed generator.
"""
NET_G = {}

NET_G['z_dim'] = 128

NET_G['shared'] = [
	('dense', (4*8), 'bn', 'relu'),
	('reshape', (4, 1, 1, 8)), 
	('transconv3d', (8, (1, 3, 1), (1, 3, 1)), 'bn', 'relu'),
	('transconv3d', (8, (1, 2, 1), (1, 2, 1)), 'bn', 'relu'),
	('transconv3d', (8, (1, 2, 1), (1, 2, 1)), 'bn', 'relu'),
	('transconv3d', (8, (1, 2, 1), (1, 2, 1)), 'bn', 'relu'),
	('transconv3d', (8, (1, 2, 1), (1, 2, 1)), 'bn', 'relu'),
	('transconv3d', (8, (1, 2, 1), (1, 2, 1)), 'bn', 'relu'),
	('transconv3d', (8, (1, 1, 2), (1, 1, 2)), 'bn', 'relu'),
	('transconv3d', (8, (1, 1, 2), (1, 1, 2)), 'bn', 'relu'),
	('transconv3d', (8, (1, 1, 3), (1, 1, 3)), 'bn', 'relu'),
	('transconv3d', (8, (1, 1, 7), (1, 1, 7)), 'bn', 'relu'),
]

NET_G['refiner'] = [
    ('identity', None, None, None),
    ('identity', None, 'bn', 'relu'),
    ('conv3d', (64, (1, 3, 12), (1, 1, 1), 'SAME'), 'bn', 'relu'),
    ('conv3d', (1, (1, 3, 12), (1, 1, 1), 'SAME'), None, None),
    ('identity', None, None, None, ('add', 0)),
    ('identity', None, 'bn', 'relu'),
    ('conv3d', (64, (1, 3, 12), (1, 1, 1), 'SAME'), 'bn', 'relu'),
    ('conv3d', (1, (1, 3, 12), (1, 1, 1), 'SAME'), None, None),
    ('identity', None, None, 'round', ('add', 4)),
]